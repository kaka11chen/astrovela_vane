//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/planning_state.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"

namespace duckdb {
namespace distributed {

//! Scoped metadata for rebuilding a transported logical plan as a distributed
//! physical plan. Catalogs may use the stable operation ID to recognize a
//! committed write before a normal create-table conflict hides their
//! extension-write provider.
class DistributedPlanningState : public ClientContextState {
public:
	static constexpr const char *KEY = "vane_distributed_planning_state";

	explicit DistributedPlanningState(string operation_id_p) : operation_id(std::move(operation_id_p)) {
		if (operation_id.empty()) {
			throw InvalidInputException("Distributed planning requires a non-empty operation ID");
		}
	}

	static shared_ptr<DistributedPlanningState> Get(ClientContext &context) {
		return context.registered_state->Get<DistributedPlanningState>(KEY);
	}

	string operation_id;
};

class ScopedDistributedPlanningState {
public:
	ScopedDistributedPlanningState(ClientContext &context_p, string operation_id)
	    : context(context_p), state(make_shared_ptr<DistributedPlanningState>(std::move(operation_id))) {
		if (DistributedPlanningState::Get(context)) {
			throw InternalException("Distributed planning state is already active");
		}
		context.registered_state->Insert(DistributedPlanningState::KEY, state);
	}

	~ScopedDistributedPlanningState() {
		context.registered_state->Remove(DistributedPlanningState::KEY);
	}

	ScopedDistributedPlanningState(const ScopedDistributedPlanningState &) = delete;
	ScopedDistributedPlanningState &operator=(const ScopedDistributedPlanningState &) = delete;

private:
	ClientContext &context;
	shared_ptr<DistributedPlanningState> state;
};

} // namespace distributed
} // namespace duckdb
