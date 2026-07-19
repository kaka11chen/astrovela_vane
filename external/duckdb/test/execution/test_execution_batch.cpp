// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/parallel/thread_context.hpp"

using namespace duckdb;

namespace {

class RetryBlockingSink : public PhysicalOperator {
public:
	explicit RetryBlockingSink(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, {}, 3) {
	}

	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &) const override {
		sink_calls++;
		if (sink_calls <= 2) {
			return SinkResultType::BLOCKED;
		}

		accepted_calls++;
		accepted_rows = chunk.size();
		accepted_values.clear();
		for (idx_t row = 0; row < chunk.size(); row++) {
			accepted_values.push_back(chunk.GetValue(0, row).GetValue<int64_t>());
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	mutable idx_t sink_calls = 0;
	mutable idx_t accepted_calls = 0;
	mutable idx_t accepted_rows = 0;
	mutable vector<int64_t> accepted_values;
};

class MultiOutputOperator : public PhysicalOperator {
public:
	explicit MultiOutputOperator(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::PROJECTION, {LogicalType::BIGINT}, 3) {
	}

	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &) const override {
		execute_calls++;
		observed_payloads.push_back(&input);
		observed_values.emplace_back();
		for (idx_t row = 0; row < input.size(); row++) {
			observed_values.back().push_back(input.GetValue(0, row).GetValue<int64_t>());
		}
		output.Reference(input);

		if (execute_calls <= 2) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		consumed_inputs++;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	mutable idx_t execute_calls = 0;
	mutable idx_t consumed_inputs = 0;
	mutable vector<const DataChunk *> observed_payloads;
	mutable vector<vector<int64_t>> observed_values;
};

static void RequireRetryablePayload(const ExecutionBatch &batch, const DataChunk *expected_payload) {
	REQUIRE(batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK);
	REQUIRE(batch.rows == 3);
	REQUIRE(batch.materialized);
	REQUIRE(batch.materialized.get() == expected_payload);
	REQUIRE(batch.materialized->size() == 3);
	REQUIRE(batch.materialized->GetValue(0, 0).GetValue<int64_t>() == 11);
	REQUIRE(batch.materialized->GetValue(0, 1).GetValue<int64_t>() == 22);
	REQUIRE(batch.materialized->GetValue(0, 2).GetValue<int64_t>() == 33);
}

static void VerifyStreamingBackpressure(idx_t threads) {
	DuckDB db(nullptr);
	Connection con(db);

	auto setting_result = con.Query("SET streaming_buffer_size='32KB'");
	REQUIRE_FALSE(setting_result->HasError());
	setting_result = con.Query("SET threads=" + to_string(threads));
	REQUIRE_FALSE(setting_result->HasError());

	auto result = con.SendQuery(R"(
		SELECT
			i,
			repeat(chr(65 + (i % 26)::INTEGER), 256) || ':' || i::VARCHAR AS payload,
			CASE WHEN i % 17 = 0 THEN NULL ELSE i * 3 END AS nullable
		FROM range(20000) t(i)
	)");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->type == QueryResultType::STREAM_RESULT);

	idx_t expected_row = 0;
	while (auto chunk = result->Fetch()) {
		REQUIRE(chunk->size() > 0);
		for (idx_t row = 0; row < chunk->size(); row++) {
			REQUIRE(chunk->GetValue(0, row).GetValue<int64_t>() == NumericCast<int64_t>(expected_row));

			string expected_payload(256, static_cast<char>('A' + expected_row % 26));
			expected_payload += ":" + to_string(expected_row);
			REQUIRE(chunk->GetValue(1, row).GetValue<string>() == expected_payload);

			auto nullable = chunk->GetValue(2, row);
			if (expected_row % 17 == 0) {
				REQUIRE(nullable.IsNull());
			} else {
				REQUIRE(nullable.GetValue<int64_t>() == NumericCast<int64_t>(expected_row * 3));
			}
			expected_row++;
		}
	}

	REQUIRE_FALSE(result->HasError());
	REQUIRE(expected_row == 20000);
}

} // namespace

TEST_CASE("ExecutionBatch payload remains retryable after a sink blocks", "[execution_batch][sink]") {
	DuckDB db(nullptr);
	Connection con(db);
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	RetryBlockingSink sink(physical_plan);

	auto payload = make_uniq<DataChunk>();
	payload->Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
	payload->SetCardinality(3);
	payload->SetValue(0, 0, Value::BIGINT(11));
	payload->SetValue(0, 1, Value::BIGINT(22));
	payload->SetValue(0, 2, Value::BIGINT(33));
	auto payload_ptr = payload.get();

	ExecutionBatch batch;
	batch.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	batch.rows = payload->size();
	batch.estimated_bytes = payload->GetAllocationSize();
	batch.materialized = std::move(payload);

	GlobalSinkState global_state;
	LocalSinkState local_state;
	InterruptState interrupt_state;
	OperatorSinkInput input {global_state, local_state, interrupt_state};

	REQUIRE(sink.SinkBatch(context, batch, input) == SinkResultType::BLOCKED);
	RequireRetryablePayload(batch, payload_ptr);
	REQUIRE(sink.accepted_calls == 0);

	REQUIRE(sink.SinkBatch(context, batch, input) == SinkResultType::BLOCKED);
	RequireRetryablePayload(batch, payload_ptr);
	REQUIRE(sink.accepted_calls == 0);

	REQUIRE(sink.SinkBatch(context, batch, input) == SinkResultType::NEED_MORE_INPUT);
	RequireRetryablePayload(batch, payload_ptr);
	REQUIRE(sink.sink_calls == 3);
	REQUIRE(sink.accepted_calls == 1);
	REQUIRE(sink.accepted_rows == 3);
	REQUIRE(sink.accepted_values == vector<int64_t> {11, 22, 33});
}

TEST_CASE("ExecutionBatch payload remains stable while an operator has more output", "[execution_batch][operator]") {
	DuckDB db(nullptr);
	Connection con(db);
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	MultiOutputOperator multi_output(physical_plan);

	auto payload = make_uniq<DataChunk>();
	payload->Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
	payload->SetCardinality(3);
	payload->SetValue(0, 0, Value::BIGINT(11));
	payload->SetValue(0, 1, Value::BIGINT(22));
	payload->SetValue(0, 2, Value::BIGINT(33));
	auto payload_ptr = payload.get();

	ExecutionBatch input;
	input.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	input.rows = payload->size();
	input.estimated_bytes = payload->GetAllocationSize();
	input.materialized = std::move(payload);

	ExecutionBatch output;
	GlobalOperatorState global_state;
	OperatorState operator_state;
	for (idx_t call = 0; call < 3; call++) {
		auto result = multi_output.ExecuteBatch(context, input, output, global_state, operator_state);
		if (call < 2) {
			REQUIRE(result == OperatorResultType::HAVE_MORE_OUTPUT);
		} else {
			REQUIRE(result == OperatorResultType::NEED_MORE_INPUT);
		}

		RequireRetryablePayload(input, payload_ptr);
		REQUIRE(output.kind == ExecutionBatchKind::MATERIALIZED_CHUNK);
		REQUIRE(output.rows == 3);
		REQUIRE(output.materialized);
		REQUIRE(output.materialized->GetValue(0, 0).GetValue<int64_t>() == 11);
		REQUIRE(output.materialized->GetValue(0, 1).GetValue<int64_t>() == 22);
		REQUIRE(output.materialized->GetValue(0, 2).GetValue<int64_t>() == 33);
	}

	REQUIRE(multi_output.execute_calls == 3);
	REQUIRE(multi_output.consumed_inputs == 1);
	REQUIRE(multi_output.observed_payloads.size() == 3);
	REQUIRE(multi_output.observed_values.size() == 3);
	for (idx_t call = 0; call < 3; call++) {
		REQUIRE(multi_output.observed_payloads[call] == payload_ptr);
		REQUIRE(multi_output.observed_values[call] == vector<int64_t> {11, 22, 33});
	}
}

TEST_CASE("Native streaming preserves ExecutionBatch payload through backpressure", "[execution_batch][streaming]") {
	SECTION("single-threaded") {
		VerifyStreamingBackpressure(1);
	}
	SECTION("multi-threaded") {
		VerifyStreamingBackpressure(4);
	}
}
