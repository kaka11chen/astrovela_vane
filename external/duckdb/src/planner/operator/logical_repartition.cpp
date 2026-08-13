// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_repartition.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/planner/operator/logical_repartition.hpp"

namespace duckdb {

LogicalRepartition::LogicalRepartition(std::shared_ptr<RepartitionSpec> repartition_spec_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_REPARTITION), repartition_spec(std::move(repartition_spec_p)) {
}

vector<ColumnBinding> LogicalRepartition::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalRepartition::EstimateCardinality(ClientContext &context) {
	return children[0]->EstimateCardinality(context);
}

void LogicalRepartition::ResolveTypes() {
	types = children[0]->types;
}

void LogicalRepartition::Serialize(Serializer &serializer) const {
	LogicalOperator::Serialize(serializer);
	auto repartition_type = repartition_spec ? static_cast<uint8_t>(repartition_spec->type())
	                                         : static_cast<uint8_t>(RepartitionSpec::Type::Random);
	serializer.WriteProperty<uint8_t>(200, "repartition_type", repartition_type);

	optional_idx num_partitions;
	vector<unique_ptr<Expression>> partition_by;
	if (!expressions.empty()) {
		partition_by.reserve(expressions.size());
		for (auto &expr : expressions) {
			partition_by.push_back(expr->Copy());
		}
	}

	if (repartition_spec) {
		switch (static_cast<RepartitionSpec::Type>(repartition_type)) {
		case RepartitionSpec::Type::Hash: {
			auto *hash_spec = dynamic_cast<HashRepartitionSpec *>(repartition_spec.get());
			if (!hash_spec) {
				throw InternalException("Expected HashRepartitionSpec for LOGICAL_REPARTITION");
			}
			auto &config = hash_spec->config();
			if (config->num_partitions) {
				num_partitions = optional_idx(static_cast<idx_t>(config->num_partitions));
			}
			break;
		}
		case RepartitionSpec::Type::Random: {
			auto *random_spec = dynamic_cast<RandomRepartitionSpec *>(repartition_spec.get());
			if (!random_spec) {
				throw InternalException("Expected RandomRepartitionSpec for LOGICAL_REPARTITION");
			}
			auto &config = random_spec->config();
			if (config->num_partitions) {
				num_partitions = optional_idx(static_cast<idx_t>(config->num_partitions));
			}
			break;
		}
		case RepartitionSpec::Type::IntoPartitions: {
			auto *into_spec = dynamic_cast<IntoPartitionsRepartitionSpec *>(repartition_spec.get());
			if (!into_spec) {
				throw InternalException("Expected IntoPartitionsRepartitionSpec for LOGICAL_REPARTITION");
			}
			auto &config = into_spec->config();
			num_partitions = optional_idx(static_cast<idx_t>(config->num_partitions));
			break;
		}
		case RepartitionSpec::Type::Range:
			throw NotImplementedException("Range repartition serialization is not supported");
		}
	}

	serializer.WritePropertyWithDefault<optional_idx>(201, "num_partitions", num_partitions);
	serializer.WritePropertyWithDefault<vector<unique_ptr<Expression>>>(202, "partition_by", partition_by);
}

unique_ptr<LogicalOperator> LogicalRepartition::Deserialize(Deserializer &deserializer) {
	auto repartition_type_val = deserializer.ReadProperty<uint8_t>(200, "repartition_type");
	auto num_partitions = deserializer.ReadPropertyWithDefault<optional_idx>(201, "num_partitions");
	auto partition_by = deserializer.ReadPropertyWithDefault<vector<unique_ptr<Expression>>>(202, "partition_by");

	size_t num_partitions_val = 0;
	if (num_partitions.IsValid()) {
		num_partitions_val = static_cast<size_t>(num_partitions.GetIndex());
	}
	vector<unique_ptr<Expression>> bound_partition_exprs;
	bound_partition_exprs.reserve(partition_by.size());
	for (auto &expr : partition_by) {
		bound_partition_exprs.push_back(expr->Copy());
	}

	std::shared_ptr<RepartitionSpec> spec;
	auto repartition_type = static_cast<RepartitionSpec::Type>(repartition_type_val);
	switch (repartition_type) {
	case RepartitionSpec::Type::Hash: {
		vector<ExprRef> expr_refs;
		expr_refs.reserve(partition_by.size());
		for (auto &expr : partition_by) {
			expr_refs.emplace_back(expr.release());
		}
		spec = RepartitionSpec::create_hash(num_partitions_val, std::move(expr_refs));
		break;
	}
	case RepartitionSpec::Type::Random:
		spec = RepartitionSpec::create_random(num_partitions_val);
		break;
	case RepartitionSpec::Type::IntoPartitions:
		if (!num_partitions_val) {
			throw InternalException("Missing num_partitions for IntoPartitions repartition");
		}
		spec = RepartitionSpec::create_into_partitions(num_partitions_val);
		break;
	case RepartitionSpec::Type::Range:
		throw NotImplementedException("Range repartition deserialization is not supported");
	default:
		throw InternalException("Unknown repartition type for LOGICAL_REPARTITION");
	}

	auto result = make_uniq<LogicalRepartition>(std::move(spec));
	result->expressions = std::move(bound_partition_exprs);
	return std::move(result);
}

} // namespace duckdb
