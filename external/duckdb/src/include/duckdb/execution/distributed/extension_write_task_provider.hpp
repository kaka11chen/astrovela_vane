// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/copy_to_file.hpp"
#include "duckdb/function/distributed_write.hpp"

namespace duckdb {

class ClientContext;

namespace distributed {

//! Stable coordinator identity for one distributed write operation. It remains
//! available even when no worker result envelope was produced, so a provider
//! can reconcile or abort the complete speculative artifact namespace.
struct DistributedWriteOperationContext {
	string operation_id;

	DUCKDB_API void Validate() const;
};

//! Dynamic coordinator state supplied by one physical extension root. Static
//! mode, protocol, and codec information is resolved from the registered write
//! operator contract and cannot be overridden by a plan.
struct DistributedExtensionWritePlan {
	string extension_name;
	string operator_name;
	string worker_bind_data;

	DUCKDB_API void Validate() const;
};

//! Coordinator-side half of the explicit distributed extension write contract.
//! The ordinary extension operator remains authoritative for native DuckDB
//! execution. Vane replaces it only for Ray execution according to WritePlan().
class ExtensionWriteTaskProvider {
public:
	virtual ~ExtensionWriteTaskProvider() = default;

	//! Immutable extension/operator key and extension-owned worker bind envelope.
	virtual const DistributedExtensionWritePlan &WritePlan() const = 0;

	//! Expose a side-effect-free, serializable worker subtree before any
	//! distributed translation pass. Resource planning can translate a physical
	//! write before Vane opens the transaction used for execution, so this hook
	//! must not create catalog entries or durable artifacts. It may only perform
	//! idempotent in-memory plan rewrites.
	virtual void PrepareDistributedWorkerPlan(ClientContext &context,
	                                          const DistributedWriteOperationContext &operation) {
	}

	//! Prepare the coordinator-owned physical write before Vane translates its
	//! worker subtree. Extensions can use this hook to resolve catalog state and
	//! replace coordinator-only children with a serializable worker input. Any
	//! durable state created here must be removable through AbortDistributedWrite.
	virtual void PrepareDistributedWrite(ClientContext &context, const DistributedWriteOperationContext &operation) {
	}

	//! Reconcile a repeated stable operation before coordinator preparation or
	//! worker execution. A valid row count is a durable proof that the catalog
	//! commit already completed; Vane returns it as a committed replay without
	//! invoking any preparation, validation, worker, finalization, or abort hook.
	//! The default means that no committed operation was found. Implementations
	//! must keep this probe side-effect free and must not use process-local state
	//! as proof of a catalog commit.
	virtual optional_idx ReconcileCommittedDistributedWrite(ClientContext &context,
	                                                        const DistributedWriteOperationContext &operation) const {
		return optional_idx();
	}

	//! Validate current coordinator/catalog state before any worker callback can
	//! run. Durable replay detection belongs in
	//! ReconcileCommittedDistributedWrite; this hook must not commit the
	//! caller-owned transaction.
	virtual void ValidateDistributedWrite(ClientContext &context,
	                                      const DistributedWriteOperationContext &operation) const = 0;

	//! Register exactly the selected task results in the active coordinator
	//! catalog transaction. Repeating finalization in the same attempt must be
	//! idempotent; a later attempt with a lost commit response is handled by
	//! durable reconciliation before this hook is reached. Returns the number of
	//! affected rows represented by the selected results.
	virtual idx_t FinalizeDistributedWrite(ClientContext &context, const DistributedWriteOperationContext &operation,
	                                       const vector<DistributedWriteTaskResult> &results) const = 0;

	//! Release commit fences or other auxiliary state only after the owner of the
	//! coordinator transaction has received definitive commit success. Durable
	//! data artifacts referenced by the committed catalog state must be retained.
	//! This hook is also invoked for a reconciled committed replay, allowing a
	//! prior attempt whose commit response was lost to finish its safe cleanup.
	//! The context owns a fresh active transaction for filesystem, secret, and
	//! catalog services; it is never the transaction that committed the write.
	virtual void ConfirmDistributedWriteCommit(ClientContext &context,
	                                           const DistributedWriteOperationContext &operation,
	                                           const vector<DistributedWriteTaskResult> &selected_results) {
	}

	//! Remove every worker artifact belonging to this operation after a known
	//! pre-commit failure. The provider must be able to clean the operation from
	//! coordinator state even when no worker envelope was returned.
	virtual void AbortDistributedWrite(ClientContext &context, const DistributedWriteOperationContext &operation,
	                                   const vector<DistributedWriteTaskResult> &selected_results) const = 0;
};

//! Coordinator-visible result of an extension write. Callback writes carry
//! only extension-owned task envelopes; file-artifact writes additionally
//! carry DuckDB's explicit output publication lifecycle.
struct DistributedExtensionWriteResult {
	DistributedExtensionWriteInfo info;
	vector<DistributedWriteTaskResult> selected_task_results;
	DistributedCopyResult file_result;
	idx_t rows_written = 0;
	idx_t bytes_written = 0;
	bool catalog_committed = false;
	bool outcome_unknown = false;
	string outcome_error;
};

//! Fixed file adapter used by FILE_ARTIFACT writes.
static constexpr const char *DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC = "duckdb.written-file-statistics";
static constexpr idx_t DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION = 1;

//! Resolve the complete immutable worker protocol from the database-local
//! concrete write registration and the physical operator's dynamic plan.
DUCKDB_API DistributedExtensionWriteInfo
ResolveDistributedExtensionWriteInfo(ClientContext &context, const DistributedExtensionWritePlan &plan);

DUCKDB_API vector<DistributedWriteTaskResult>
EncodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info,
                                  const DistributedWriteOperationContext &operation,
                                  const vector<DistributedCopyFileInfo> &files);
DUCKDB_API vector<DistributedCopyFileInfo>
DecodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info,
                                  const DistributedWriteOperationContext &operation,
                                  const vector<DistributedWriteTaskResult> &results);

//! Decode the one-column BLOB output produced by
//! PhysicalDistributedExtensionWrite and enforce the coordinator plan's exact
//! capability and fragment codec.
DUCKDB_API vector<DistributedWriteTaskResult>
ParseDistributedWriteTaskResults(const DistributedExtensionWriteInfo &info,
                                 const DistributedWriteOperationContext &operation,
                                 const vector<ResultPartitionRef> &partitions);

} // namespace distributed
} // namespace duckdb
