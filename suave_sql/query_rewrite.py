import ast
import json
import re
from pathlib import Path

CLASS_NAME = "Queries"  # Change to your class name
INPUT_FILE = "sql_funcs.py"

OUTPUT_DIR = Path("queries")
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_FILE = "migration_report.json"

# Regex to catch all {…} in f-strings
FSTRING_EXPR = re.compile(r"{([^}]+)}")



def sanitize_param_name(expr: str) -> str:
    """Convert any expression to a safe Python identifier for bind parameters."""
    expr = expr.strip()
    expr = expr.replace("self.", "")
    expr = re.sub(r"\W+", "_", expr)
    expr = re.sub(r"__+", "_", expr)
    expr = expr.strip("_")
    return expr


def normalize_sql(sql: str, param_map: dict):
    """
    Replace all {…} in SQL with :param_name and record mapping
    """
    def repl(match):
        expr = match.group(1)
        param_name = sanitize_param_name(expr)
        param_map[param_name] = expr
        return f":{param_name}"

    return FSTRING_EXPR.sub(repl, sql)


def resolve_node(node, assignments, seen=None):
    """Recursively resolve AST nodes into strings, following variable assignments."""
    if seen is None:
        seen = set()
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Str):  # Python <3.8
        return node.s
    if isinstance(node, ast.Name):
        if node.id in seen:
            return "{" + node.id + "}"  # circular reference fallback
        if node.id in assignments:
            seen.add(node.id)
            return resolve_node(assignments[node.id], assignments, seen)
        return "{" + node.id + "}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = resolve_node(node.left, assignments, seen.copy())
        right = resolve_node(node.right, assignments, seen.copy())
        return left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, (ast.Str, ast.Constant)):
                parts.append(v.s if hasattr(v, "s") else v.value)
            elif isinstance(v, ast.FormattedValue):
                parts.append("{" + ast.unparse(v.value) + "}")
        return "".join(parts)
    return ""


def extract_assignments_and_branches(method):
    assignments = {}
    branches = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node.value
        elif isinstance(node, ast.If):
            cond = ast.unparse(node.test)
            for branch, key in [(node.body, "true"), (node.orelse, "false")]:
                for stmt in branch:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                        var = stmt.targets[0].id
                        if isinstance(stmt.value, (ast.Constant, ast.Str, ast.JoinedStr)):
                            branches.setdefault(var, {})[key] = stmt.value
                            branches[var]["condition"] = cond
    return assignments, branches


def merge_branches(sql, branches):
    """
    Convert simple branch assignments into CASE statements in SQL
    """
    for var, info in branches.items():
        cond = info.get("condition")
        true_val = resolve_node(info.get("true"), {}) if info.get("true") else None
        false_val = resolve_node(info.get("false"), {}) if info.get("false") else None
        if true_val and false_val:
            sql = re.sub(rf"\b{var}\b", f"CASE WHEN :{cond} THEN {true_val} ELSE {false_val} END", sql)
    return sql


def extract_query(method):
    """
    Recursively resolve SQL passed to query_run(), preserve docstring and detect parameters
    """
    # full multi-line docstring
    docstring_node = method.body[0]
    docstring = ""
    if isinstance(docstring_node, ast.Expr) and isinstance(docstring_node.value, (ast.Str, ast.Constant)):
        docstring = docstring_node.value.s if hasattr(docstring_node.value, "s") else docstring_node.value.value
        docstring = docstring.strip()

    assignments, branches = extract_assignments_and_branches(method)
    param_map = {}

    sql = None
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "query_run":
                arg = node.args[0]
                # Recursively resolve variable references
                resolved_sql = resolve_node(arg, assignments)
                resolved_sql = merge_branches(resolved_sql, branches)
                # Convert all {…} to safe parameters
                sql = normalize_sql(resolved_sql, param_map)
                break

    return sql, docstring, param_map


def migrate(file):
    with open(file) as f:
        tree = ast.parse(f.read())

    report = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME:
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                name = method.name
                sql, docstring, param_map = extract_query(method)
                entry = {}
                if sql:
                    path = OUTPUT_DIR / f"{name}.sql"
                    with open(path, "w") as f:
                        if docstring:
                            # preserve multi-line hierarchical docstring
                            for line in docstring.splitlines():
                                f.write(f"-- {line.rstrip()}\n")
                            f.write("\n")
                        f.write(sql)
                    entry["sql_file"] = str(path)
                    entry["parameters"] = param_map
                else:
                    entry["status"] = "no query detected"
                report[name] = entry

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("Migration complete. SQL files written to", OUTPUT_DIR)
    print("Report written to", REPORT_FILE)
# -------------------------
# Run migration
# -------------------------
if __name__ == "__main__":
    migrate(INPUT_FILE)