// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/main/relation/aggregate_relation.hpp"
#include "duckdb/main/relation/cross_product_relation.hpp"
#include "duckdb/main/relation/distinct_relation.hpp"
#include "duckdb/main/relation/explain_relation.hpp"
#include "duckdb/main/relation/filter_relation.hpp"
#include "duckdb/main/relation/insert_relation.hpp"
#include "duckdb/main/relation/limit_relation.hpp"
#include "duckdb/main/relation/repartition_relation.hpp"
#include "duckdb/main/relation/local_exchange_relation.hpp"
#include "duckdb/main/relation/order_relation.hpp"
#include "duckdb/main/relation/projection_relation.hpp"
#include "duckdb/main/relation/setop_relation.hpp"
#include "duckdb/main/relation/subquery_relation.hpp"
#include "duckdb/main/relation/table_function_relation.hpp"
#include "duckdb/main/relation/create_table_relation.hpp"
#include "duckdb/main/relation/create_view_relation.hpp"
#include "duckdb/main/relation/write_csv_relation.hpp"
#include "duckdb/main/relation/write_parquet_relation.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/parser/tableref/bound_ref_wrapper.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/tableref/list.hpp"
#include "duckdb/main/relation/join_relation.hpp"
#include "duckdb/main/relation/value_relation.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"

namespace duckdb {

shared_ptr<Relation> Relation::Project(const string &select_list) {
	return Project(select_list, vector<string>());
}

shared_ptr<Relation> Relation::Project(const string &expression, const string &alias) {
	return Project(expression, vector<string>({alias}));
}

shared_ptr<Relation> Relation::Project(const string &select_list, const vector<string> &aliases) {
	auto expressions = Parser::ParseExpressionList(select_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<ProjectionRelation>(shared_from_this(), std::move(expressions), aliases);
}

shared_ptr<Relation> Relation::Project(const vector<string> &expressions) {
	vector<string> aliases;
	return Project(expressions, aliases);
}

shared_ptr<Relation> Relation::Project(vector<unique_ptr<ParsedExpression>> expressions,
                                       const vector<string> &aliases) {
	return make_shared_ptr<ProjectionRelation>(shared_from_this(), std::move(expressions), aliases);
}

static vector<unique_ptr<ParsedExpression>> StringListToExpressionList(const ClientContext &context,
                                                                       const vector<string> &expressions) {
	if (expressions.empty()) {
		throw ParserException("Zero expressions provided");
	}
	vector<unique_ptr<ParsedExpression>> result_list;
	for (auto &expr : expressions) {
		auto expression_list = Parser::ParseExpressionList(expr, context.GetParserOptions());
		if (expression_list.size() != 1) {
			throw ParserException("Expected a single expression in the expression list");
		}
		result_list.push_back(std::move(expression_list[0]));
	}
	return result_list;
}

shared_ptr<Relation> Relation::Project(const vector<string> &expressions, const vector<string> &aliases) {
	auto result_list = StringListToExpressionList(*context->GetContext(), expressions);
	return make_shared_ptr<ProjectionRelation>(shared_from_this(), std::move(result_list), aliases);
}

shared_ptr<Relation> Relation::Filter(const string &expression) {
	auto expression_list = Parser::ParseExpressionList(expression, context->GetContext()->GetParserOptions());
	if (expression_list.size() != 1) {
		throw ParserException("Expected a single expression as filter condition");
	}
	return Filter(std::move(expression_list[0]));
}

shared_ptr<Relation> Relation::Filter(unique_ptr<ParsedExpression> expression) {
	return make_shared_ptr<FilterRelation>(shared_from_this(), std::move(expression));
}

shared_ptr<Relation> Relation::Filter(const vector<string> &expressions) {
	// if there are multiple expressions, we AND them together
	auto expression_list = StringListToExpressionList(*context->GetContext(), expressions);
	D_ASSERT(!expression_list.empty());

	auto expr = std::move(expression_list[0]);
	for (idx_t i = 1; i < expression_list.size(); i++) {
		expr = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(expr),
		                                        std::move(expression_list[i]));
	}
	return make_shared_ptr<FilterRelation>(shared_from_this(), std::move(expr));
}

shared_ptr<Relation> Relation::Limit(int64_t limit, int64_t offset) {
	return make_shared_ptr<LimitRelation>(shared_from_this(), limit, offset);
}

shared_ptr<Relation> Relation::Repartition(idx_t num_partitions, vector<unique_ptr<ParsedExpression>> partition_by) {
	if (num_partitions > 0 && false) {
		throw InvalidInputException("num_partitions must be greater than zero");
	}
	return make_shared_ptr<RepartitionRelation>(shared_from_this(), num_partitions, std::move(partition_by));
}

shared_ptr<Relation> Relation::Repartition(idx_t num_partitions, const vector<string> &partition_by) {
	vector<unique_ptr<ParsedExpression>> expressions;
	if (!partition_by.empty()) {
		expressions = StringListToExpressionList(*context->GetContext(), partition_by);
	}
	return Repartition(num_partitions, std::move(expressions));
}

shared_ptr<Relation> Relation::LocalExchange(idx_t num_partitions) {
	return make_shared_ptr<LocalExchangeRelation>(shared_from_this(), num_partitions);
}

shared_ptr<Relation> Relation::Order(const string &expression) {
	auto order_list = Parser::ParseOrderList(expression, context->GetContext()->GetParserOptions());
	return Order(std::move(order_list));
}

shared_ptr<Relation> Relation::Order(vector<OrderByNode> order_list) {
	return make_shared_ptr<OrderRelation>(shared_from_this(), std::move(order_list));
}

shared_ptr<Relation> Relation::Order(const vector<string> &expressions) {
	if (expressions.empty()) {
		throw ParserException("Zero ORDER BY expressions provided");
	}
	vector<OrderByNode> order_list;
	for (auto &expression : expressions) {
		auto inner_list = Parser::ParseOrderList(expression, context->GetContext()->GetParserOptions());
		if (inner_list.size() != 1) {
			throw ParserException("Expected a single ORDER BY expression in the expression list");
		}
		order_list.push_back(std::move(inner_list[0]));
	}
	return Order(std::move(order_list));
}

shared_ptr<Relation> Relation::Join(const shared_ptr<Relation> &other, const string &condition, JoinType type,
                                    JoinRefType ref_type) {
	auto expression_list = Parser::ParseExpressionList(condition, context->GetContext()->GetParserOptions());
	D_ASSERT(!expression_list.empty());
	return Join(other, std::move(expression_list), type, ref_type);
}

shared_ptr<Relation> Relation::Join(const shared_ptr<Relation> &other,
                                    vector<unique_ptr<ParsedExpression>> expression_list, JoinType type,
                                    JoinRefType ref_type) {
	if (expression_list.size() > 1 || expression_list[0]->GetExpressionType() == ExpressionType::COLUMN_REF) {
		// multiple columns or single column ref: the condition is a USING list
		vector<string> using_columns;
		for (auto &expr : expression_list) {
			if (expr->GetExpressionType() != ExpressionType::COLUMN_REF) {
				throw ParserException("Expected a single expression as join condition");
			}
			auto &colref = expr->Cast<ColumnRefExpression>();
			if (colref.IsQualified()) {
				throw ParserException("Expected unqualified column for column in USING clause");
			}
			using_columns.push_back(colref.column_names[0]);
		}
		return make_shared_ptr<JoinRelation>(shared_from_this(), other, std::move(using_columns), type, ref_type);
	} else {
		// single expression that is not a column reference: use the expression as a join condition
		return make_shared_ptr<JoinRelation>(shared_from_this(), other, std::move(expression_list[0]), type, ref_type);
	}
}

shared_ptr<Relation> Relation::CrossProduct(const shared_ptr<Relation> &other, JoinRefType join_ref_type) {
	return make_shared_ptr<CrossProductRelation>(shared_from_this(), other, join_ref_type);
}

shared_ptr<Relation> Relation::Union(const shared_ptr<Relation> &other) {
	return make_shared_ptr<SetOpRelation>(shared_from_this(), other, SetOperationType::UNION, true);
}

shared_ptr<Relation> Relation::Except(const shared_ptr<Relation> &other) {
	return make_shared_ptr<SetOpRelation>(shared_from_this(), other, SetOperationType::EXCEPT, true);
}

shared_ptr<Relation> Relation::Intersect(const shared_ptr<Relation> &other) {
	return make_shared_ptr<SetOpRelation>(shared_from_this(), other, SetOperationType::INTERSECT, true);
}

shared_ptr<Relation> Relation::Distinct() {
	return make_shared_ptr<DistinctRelation>(shared_from_this());
}

shared_ptr<Relation> Relation::Alias(const string &alias) {
	return make_shared_ptr<SubqueryRelation>(shared_from_this(), alias);
}

shared_ptr<Relation> Relation::Aggregate(const string &aggregate_list) {
	auto expression_list = Parser::ParseExpressionList(aggregate_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expression_list));
}

shared_ptr<Relation> Relation::Aggregate(vector<unique_ptr<ParsedExpression>> expressions) {
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expressions));
}

shared_ptr<Relation> Relation::Aggregate(const string &aggregate_list, const string &group_list) {
	auto expression_list = Parser::ParseExpressionList(aggregate_list, context->GetContext()->GetParserOptions());
	auto groups = Parser::ParseGroupByList(group_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expression_list), std::move(groups));
}

shared_ptr<Relation> Relation::Aggregate(const vector<string> &aggregates) {
	auto aggregate_list = StringListToExpressionList(*context->GetContext(), aggregates);
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(aggregate_list));
}

shared_ptr<Relation> Relation::Aggregate(const vector<string> &aggregates, const vector<string> &groups) {
	auto aggregate_list = StringUtil::Join(aggregates, ", ");
	auto group_list = StringUtil::Join(groups, ", ");
	return this->Aggregate(aggregate_list, group_list);
}

shared_ptr<Relation> Relation::Aggregate(vector<unique_ptr<ParsedExpression>> expressions, const string &group_list) {
	auto groups = Parser::ParseGroupByList(group_list, context->GetContext()->GetParserOptions());
	return make_shared_ptr<AggregateRelation>(shared_from_this(), std::move(expressions), std::move(groups));
}

string Relation::GetAlias() {
	return alias;
}

unique_ptr<TableRef> Relation::GetTableRef() {
	if (!CanSerializeToQueryNode()) {
		throw NotImplementedException("Cannot create a table reference for a relation that cannot be faithfully "
		                              "represented as a SQL query node; conversion would discard the exchange or lose "
		                              "relation bindings");
	}
	auto select = make_uniq<SelectStatement>();
	select->node = GetQueryNode();
	return make_uniq<SubqueryRef>(std::move(select), GetAlias());
}

unique_ptr<QueryResult> Relation::Execute() {
	return context->GetContext()->Execute(shared_from_this());
}

unique_ptr<QueryResult> Relation::ExecuteOrThrow() {
	auto res = Execute();
	D_ASSERT(res);
	if (res->HasError()) {
		res->ThrowError();
	}
	return res;
}

BoundStatement Relation::Bind(Binder &binder) {
	if (!CanSerializeToQueryNode()) {
		throw NotImplementedException(
		    "Cannot bind a relation that cannot be faithfully represented as a SQL query node; "
		    "conversion would discard the exchange or lose relation bindings");
	}
	SelectStatement stmt;
	stmt.node = GetQueryNode();
	return binder.Bind(stmt.Cast<SQLStatement>());
}

BoundStatement Relation::BindAsInput(Binder &binder) {
	return Bind(binder);
}

bool Relation::RequiresDirectRelationBinding(Relation &child) {
	if (!child.CanSerializeToQueryNode()) {
		return child.CanBindAsInput();
	}
	return ExposesMultiSourceBindings(child);
}

bool Relation::ExposesMultiSourceBindings(Relation &child) {
	auto child_ptr = &child;
	while (child_ptr->InheritsColumnBindings()) {
		child_ptr = child_ptr->ChildRelation();
		D_ASSERT(child_ptr);
	}
	return child_ptr->type == RelationType::JOIN_RELATION || child_ptr->type == RelationType::CROSS_PRODUCT_RELATION;
}

bool Relation::RequiresSQLMultiSourceBinding(Relation &child) {
	auto child_ptr = &child;
	// Filters can be merged into the same SELECT without changing relational
	// ordering. Other inheriting relations (LIMIT, DISTINCT, ORDER, etc.) are
	// semantic boundaries and must remain separate plan nodes.
	while (child_ptr->type == RelationType::FILTER_RELATION) {
		child_ptr = child_ptr->ChildRelation();
	}
	return child_ptr->type == RelationType::JOIN_RELATION || child_ptr->type == RelationType::CROSS_PRODUCT_RELATION;
}

namespace {

class SerializedExpressionScopeChecker {
public:
	explicit SerializedExpressionScopeChecker(string child_alias_p) : child_alias(std::move(child_alias_p)) {
	}

	bool Check(const ParsedExpression &expression) {
		VisitExpression(expression);
		return serializable;
	}

private:
	bool QualifierIsVisible(const string &qualifier) const {
		if (StringUtil::CIEquals(qualifier, child_alias)) {
			return true;
		}
		for (auto scope = query_scopes.rbegin(); scope != query_scopes.rend(); scope++) {
			for (auto &alias : *scope) {
				if (StringUtil::CIEquals(qualifier, alias)) {
					return true;
				}
			}
		}
		return false;
	}

	void VisitExpression(const ParsedExpression &expression) {
		if (!serializable) {
			return;
		}
		switch (expression.GetExpressionClass()) {
		case ExpressionClass::COLUMN_REF: {
			auto &column_ref = expression.Cast<ColumnRefExpression>();
			if (column_ref.IsQualified() && !QualifierIsVisible(column_ref.GetTableName())) {
				serializable = false;
			}
			return;
		}
		case ExpressionClass::STAR: {
			auto &star = expression.Cast<StarExpression>();
			auto references_hidden_table = [&](const QualifiedColumnName &column) {
				return column.IsQualified() && !QualifierIsVisible(column.table);
			};
			if ((!star.relation_name.empty() && !QualifierIsVisible(star.relation_name)) ||
			    std::any_of(star.exclude_list.begin(), star.exclude_list.end(), references_hidden_table) ||
			    std::any_of(star.rename_list.begin(), star.rename_list.end(),
			                [&](const auto &entry) { return references_hidden_table(entry.first); })) {
				serializable = false;
				return;
			}
			break;
		}
		case ExpressionClass::SUBQUERY: {
			auto &subquery = expression.Cast<SubqueryExpression>();
			if (subquery.child) {
				VisitExpression(*subquery.child);
			}
			if (!subquery.subquery || !subquery.subquery->node) {
				serializable = false;
				return;
			}
			VisitQueryNode(*subquery.subquery->node);
			return;
		}
		default:
			break;
		}
		ParsedExpressionIterator::EnumerateChildren(expression,
		                                            [&](const ParsedExpression &child) { VisitExpression(child); });
	}

	void CollectTableAliases(const TableRef &ref, vector<string> &aliases) {
		if (!ref.alias.empty()) {
			aliases.push_back(ref.alias);
			return;
		}
		switch (ref.type) {
		case TableReferenceType::BASE_TABLE:
			aliases.push_back(ref.Cast<BaseTableRef>().table_name);
			break;
		case TableReferenceType::JOIN: {
			auto &join = ref.Cast<JoinRef>();
			CollectTableAliases(*join.left, aliases);
			CollectTableAliases(*join.right, aliases);
			break;
		}
		case TableReferenceType::TABLE_FUNCTION: {
			auto &table_function = ref.Cast<TableFunctionRef>();
			if (table_function.function && table_function.function->GetExpressionClass() == ExpressionClass::FUNCTION) {
				aliases.push_back(table_function.function->Cast<FunctionExpression>().function_name);
			}
			break;
		}
		case TableReferenceType::PIVOT:
			CollectTableAliases(*ref.Cast<PivotRef>().source, aliases);
			break;
		default:
			break;
		}
	}

	void VisitTableRef(TableRef &ref) {
		if (!serializable) {
			return;
		}
		switch (ref.type) {
		case TableReferenceType::EXPRESSION_LIST: {
			auto &expression_list = ref.Cast<ExpressionListRef>();
			for (auto &row : expression_list.values) {
				for (auto &expression : row) {
					VisitExpression(*expression);
				}
			}
			break;
		}
		case TableReferenceType::JOIN: {
			auto &join = ref.Cast<JoinRef>();
			VisitTableRef(*join.left);
			VisitTableRef(*join.right);
			if (join.condition) {
				VisitExpression(*join.condition);
			}
			for (auto &expression : join.duplicate_eliminated_columns) {
				VisitExpression(*expression);
			}
			break;
		}
		case TableReferenceType::PIVOT: {
			auto &pivot = ref.Cast<PivotRef>();
			VisitTableRef(*pivot.source);
			for (auto &aggregate : pivot.aggregates) {
				VisitExpression(*aggregate);
			}
			for (auto &column : pivot.pivots) {
				for (auto &expression : column.pivot_expressions) {
					VisitExpression(*expression);
				}
				for (auto &entry : column.entries) {
					if (entry.expr) {
						VisitExpression(*entry.expr);
					}
				}
				if (column.subquery) {
					VisitQueryNode(*column.subquery);
				}
			}
			break;
		}
		case TableReferenceType::SUBQUERY: {
			auto &subquery = ref.Cast<SubqueryRef>();
			VisitQueryNode(*subquery.subquery->node);
			break;
		}
		case TableReferenceType::TABLE_FUNCTION: {
			auto &table_function = ref.Cast<TableFunctionRef>();
			VisitExpression(*table_function.function);
			if (table_function.subquery) {
				VisitQueryNode(*table_function.subquery->node);
			}
			break;
		}
		case TableReferenceType::SHOW_REF: {
			auto &show = ref.Cast<ShowRef>();
			if (show.query) {
				VisitQueryNode(*show.query);
			}
			break;
		}
		case TableReferenceType::BASE_TABLE:
		case TableReferenceType::EMPTY_FROM:
		case TableReferenceType::COLUMN_DATA:
		case TableReferenceType::DELIM_GET:
			break;
		default:
			serializable = false;
			break;
		}
	}

	void VisitQueryNode(QueryNode &node) {
		if (!serializable) {
			return;
		}
		for (auto &entry : node.cte_map.map) {
			VisitQueryNode(*entry.second->query->node);
		}
		switch (node.type) {
		case QueryNodeType::SELECT_NODE: {
			auto &select = node.Cast<SelectNode>();
			vector<string> aliases;
			if (select.from_table) {
				CollectTableAliases(*select.from_table, aliases);
			}
			query_scopes.push_back(std::move(aliases));
			for (auto &expression : select.select_list) {
				VisitExpression(*expression);
			}
			for (auto &expression : select.groups.group_expressions) {
				VisitExpression(*expression);
			}
			if (select.where_clause) {
				VisitExpression(*select.where_clause);
			}
			if (select.having) {
				VisitExpression(*select.having);
			}
			if (select.qualify) {
				VisitExpression(*select.qualify);
			}
			if (select.from_table) {
				VisitTableRef(*select.from_table);
			}
			ParsedExpressionIterator::EnumerateQueryNodeModifiers(
			    node, [&](unique_ptr<ParsedExpression> &expression) { VisitExpression(*expression); });
			query_scopes.pop_back();
			break;
		}
		case QueryNodeType::SET_OPERATION_NODE: {
			auto &set_operation = node.Cast<SetOperationNode>();
			for (auto &child : set_operation.children) {
				VisitQueryNode(*child);
			}
			ParsedExpressionIterator::EnumerateQueryNodeModifiers(
			    node, [&](unique_ptr<ParsedExpression> &expression) { VisitExpression(*expression); });
			break;
		}
		case QueryNodeType::RECURSIVE_CTE_NODE: {
			auto &recursive_cte = node.Cast<RecursiveCTENode>();
			VisitQueryNode(*recursive_cte.left);
			VisitQueryNode(*recursive_cte.right);
			for (auto &target : recursive_cte.key_targets) {
				VisitExpression(*target);
			}
			ParsedExpressionIterator::EnumerateQueryNodeModifiers(
			    node, [&](unique_ptr<ParsedExpression> &expression) { VisitExpression(*expression); });
			break;
		}
		default:
			serializable = false;
			break;
		}
	}

private:
	string child_alias;
	vector<vector<string>> query_scopes;
	bool serializable = true;
};

} // namespace

bool Relation::CanSerializeExpressionOnChild(Relation &child, const ParsedExpression &expression) {
	if (!child.CanSerializeToQueryNode()) {
		return false;
	}
	if (!ExposesMultiSourceBindings(child) || RequiresSQLMultiSourceBinding(child)) {
		return true;
	}

	// A semantic boundary above a join is emitted as a subquery. Check every
	// qualified reference against the aliases that survive that boundary and
	// against aliases introduced by nested query scopes.
	return SerializedExpressionScopeChecker(child.GetAlias()).Check(expression);
}

unique_ptr<TableRef> Relation::BindRelationInput(Binder &binder, Relation &child) {
	// BoundRefWrapper lets the normal SELECT binder consume a non-SQL child
	// without converting its logical plan back into a QueryNode. Keep the
	// child's full BindContext for binding-preserving relations and joins.
	auto child_binder = Binder::CreateBinder(binder.context, binder.shared_from_this());
	auto child_bound = child.BindAsInput(*child_binder);
	binder.MoveCorrelatedExpressionsFrom(*child_binder);

	bool multi_source = child.type == RelationType::JOIN_RELATION || child.type == RelationType::CROSS_PRODUCT_RELATION;
	bool preserves_bindings = child.InheritsColumnBindings() || multi_source;
	if (!preserves_bindings) {
		// Projections, aggregates, and explicit aliases establish a new scope.
		// Rebase that scope on the bound plan's single output table index.
		auto input_binder = Binder::CreateBinder(binder.context, binder.shared_from_this());
		auto names = BindContext::AliasColumnNames(child.GetAlias(), child_bound.names, {});
		input_binder->bind_context.AddGenericBinding(child_bound.plan->GetRootIndex(), child.GetAlias(), names,
		                                             child_bound.types);
		child_binder = std::move(input_binder);
	}
	return make_uniq<BoundRefWrapper>(std::move(child_bound), std::move(child_binder));
}

BoundStatement Relation::BindSelectNodeOnChild(Binder &binder, Relation &child, unique_ptr<SelectNode> select_node) {
	select_node->from_table = BindRelationInput(binder, child);
	SelectStatement stmt;
	stmt.node = std::move(select_node);
	return binder.Bind(stmt.Cast<SQLStatement>());
}

unique_ptr<LogicalOperator> Relation::PlanRelationFilter(Binder &binder, unique_ptr<Expression> condition,
                                                         unique_ptr<LogicalOperator> child) {
	return binder.PlanFilter(std::move(condition), std::move(child));
}

void Relation::ExpandRelationFilter(Binder &binder, unique_ptr<ParsedExpression> &condition) {
	binder.BindWhereStarExpression(condition);
}

void Relation::ExpandRelationStar(Binder &binder, unique_ptr<ParsedExpression> expression,
                                  vector<unique_ptr<ParsedExpression>> &result) {
	binder.ExpandStarExpression(std::move(expression), result);
}

void Relation::PlanRelationSubqueries(Binder &binder, unique_ptr<Expression> &expression,
                                      unique_ptr<LogicalOperator> &root) {
	binder.PlanSubqueries(expression, root);
}

shared_ptr<Relation> Relation::InsertRel(const string &schema_name, const string &table_name) {
	return InsertRel(INVALID_CATALOG, schema_name, table_name);
}

shared_ptr<Relation> Relation::InsertRel(const string &catalog_name, const string &schema_name,
                                         const string &table_name) {
	return make_shared_ptr<InsertRelation>(shared_from_this(), catalog_name, schema_name, table_name);
}

void Relation::Insert(const string &table_name) {
	Insert(INVALID_SCHEMA, table_name);
}

void Relation::Insert(const string &schema_name, const string &table_name) {
	Insert(INVALID_CATALOG, schema_name, table_name);
}

void Relation::Insert(const string &catalog_name, const string &schema_name, const string &table_name) {
	auto insert = InsertRel(catalog_name, schema_name, table_name);
	auto res = insert->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to insert into table '" + table_name + "': ";
		res->ThrowError(prepended_message);
	}
}

void Relation::Insert(const vector<vector<Value>> &values) {
	throw InvalidInputException("INSERT with values can only be used on base tables!");
}

void Relation::Insert(vector<vector<unique_ptr<ParsedExpression>>> &&expressions) {
	(void)std::move(expressions);
	throw InvalidInputException("INSERT with expressions can only be used on base tables!");
}

shared_ptr<Relation> Relation::CreateRel(const string &schema_name, const string &table_name, bool temporary,
                                         OnCreateConflict on_conflict) {
	return CreateRel(INVALID_CATALOG, schema_name, table_name, temporary, on_conflict);
}

shared_ptr<Relation> Relation::CreateRel(const string &catalog_name, const string &schema_name,
                                         const string &table_name, bool temporary, OnCreateConflict on_conflict) {
	return make_shared_ptr<CreateTableRelation>(shared_from_this(), catalog_name, schema_name, table_name, temporary,
	                                            on_conflict);
}

void Relation::Create(const string &table_name, bool temporary, OnCreateConflict on_conflict) {
	Create(INVALID_CATALOG, INVALID_SCHEMA, table_name, temporary, on_conflict);
}

void Relation::Create(const string &schema_name, const string &table_name, bool temporary,
                      OnCreateConflict on_conflict) {
	Create(INVALID_CATALOG, schema_name, table_name, temporary, on_conflict);
}

void Relation::Create(const string &catalog_name, const string &schema_name, const string &table_name, bool temporary,
                      OnCreateConflict on_conflict) {
	if (table_name.empty()) {
		throw ParserException("Empty table name not supported");
	}
	auto create = CreateRel(catalog_name, schema_name, table_name, temporary, on_conflict);
	auto res = create->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to create table '" + table_name + "': ";
		res->ThrowError(prepended_message);
	}
}

shared_ptr<Relation> Relation::WriteCSVRel(const string &csv_file, case_insensitive_map_t<vector<Value>> options) {
	return make_shared_ptr<duckdb::WriteCSVRelation>(shared_from_this(), csv_file, std::move(options));
}

void Relation::WriteCSV(const string &csv_file, case_insensitive_map_t<vector<Value>> options) {
	auto write_csv = WriteCSVRel(csv_file, std::move(options));
	auto res = write_csv->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to write '" + csv_file + "': ";
		res->ThrowError(prepended_message);
	}
}

shared_ptr<Relation> Relation::WriteParquetRel(const string &parquet_file,
                                               case_insensitive_map_t<vector<Value>> options) {
	auto write_parquet =
	    make_shared_ptr<duckdb::WriteParquetRelation>(shared_from_this(), parquet_file, std::move(options));
	return std::move(write_parquet);
}

void Relation::WriteParquet(const string &parquet_file, case_insensitive_map_t<vector<Value>> options) {
	auto write_parquet = WriteParquetRel(parquet_file, std::move(options));
	auto res = write_parquet->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to write '" + parquet_file + "': ";
		res->ThrowError(prepended_message);
	}
}

shared_ptr<Relation> Relation::CreateView(const string &name, bool replace, bool temporary) {
	return CreateView(INVALID_SCHEMA, name, replace, temporary);
}

shared_ptr<Relation> Relation::CreateView(const string &schema_name, const string &name, bool replace, bool temporary) {
	auto view = make_shared_ptr<CreateViewRelation>(shared_from_this(), schema_name, name, replace, temporary);
	auto res = view->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to create view '" + name + "': ";
		res->ThrowError(prepended_message);
	}
	return shared_from_this();
}

unique_ptr<QueryResult> Relation::Query(const string &sql) const {
	return context->GetContext()->Query(sql, false);
}

unique_ptr<QueryResult> Relation::Query(const string &name, const string &sql) {
	bool replace = true;
	bool temp = IsReadOnly();
	CreateView(name, replace, temp);
	return Query(sql);
}

unique_ptr<QueryResult> Relation::Explain(ExplainType type, ExplainFormat format) {
	auto explain = make_shared_ptr<ExplainRelation>(shared_from_this(), type, format);
	return explain->Execute();
}

void Relation::TryBindRelation(vector<ColumnDefinition> &columns) {
	context->TryBindRelation(*this, columns);
}

void Relation::Update(const string &update, const string &condition) {
	throw InvalidInputException("UPDATE can only be used on base tables!");
}

void Relation::Update(vector<string>, // NOLINT: unused variable / copied on every invocation ...
                      vector<unique_ptr<ParsedExpression>> &&update, // NOLINT: unused variable
                      unique_ptr<ParsedExpression> condition) {      // NOLINT: unused variable
	(void)std::move(update);
	(void)std::move(condition);
	throw InvalidInputException("UPDATE can only be used on base tables!");
}

void Relation::Delete(const string &condition) {
	throw InvalidInputException("DELETE can only be used on base tables!");
}

shared_ptr<Relation> Relation::TableFunction(const std::string &fname, const vector<Value> &values,
                                             const named_parameter_map_t &named_parameters) {
	return make_shared_ptr<TableFunctionRelation>(context->GetContext(), fname, values, named_parameters,
	                                              shared_from_this());
}

shared_ptr<Relation> Relation::TableFunction(const std::string &fname, const vector<Value> &values) {
	return make_shared_ptr<TableFunctionRelation>(context->GetContext(), fname, values, shared_from_this());
}

string Relation::ToString() {
	string str;
	str += "---------------------\n";
	str += "--- Relation Tree ---\n";
	str += "---------------------\n";
	str += ToString(0);
	str += "\n\n";
	str += "---------------------\n";
	str += "-- Result Columns  --\n";
	str += "---------------------\n";
	auto &cols = Columns();
	for (idx_t i = 0; i < cols.size(); i++) {
		str += "- " + cols[i].Name() + " (" + cols[i].Type().ToString() + ")\n";
	}
	return str;
}

// LCOV_EXCL_START
string Relation::GetQuery() {
	if (!CanSerializeToQueryNode()) {
		return string();
	}
	return GetQueryNode()->ToString();
}

void Relation::Head(idx_t limit) {
	auto limit_node = Limit(NumericCast<int64_t>(limit));
	limit_node->Execute()->Print();
}
// LCOV_EXCL_STOP

void Relation::Print() {
	Printer::Print(ToString());
}

string Relation::RenderWhitespace(idx_t depth) {
	return string(depth * 2, ' ');
}

void Relation::AddExternalDependency(shared_ptr<ExternalDependency> dependency) {
	external_dependencies.push_back(std::move(dependency));
}

vector<shared_ptr<ExternalDependency>> Relation::GetAllDependencies() {
	vector<shared_ptr<ExternalDependency>> all_dependencies;
	Relation *cur = this;
	while (cur) {
		for (auto &dep : cur->external_dependencies) {
			all_dependencies.push_back(dep);
		}
		cur = cur->ChildRelation();
	}
	return all_dependencies;
}

} // namespace duckdb
