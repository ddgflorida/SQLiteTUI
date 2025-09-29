# SQLiteTUI - Browse SQLite dateabases
import sys
import sqlite3
import re
import json
from pathlib import Path
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Header,
    Footer,
    Tree,
    Static,
    TextArea,
    Button,
    TabbedContent,
    TabPane,
)
from textual import on

# --- History Configuration ---
HISTORY_FILE = Path.home() / ".sqlite_browser_history.json"
MAX_HISTORY_ITEMS = 100

# --- Helper function for SQL identifier quoting ---

SQLITE_KEYWORDS = {
    "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ALWAYS", "ANALYZE", "AND",
    "AS", "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN", "BY",
    "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN", "COMMIT", "CONFLICT",
    "CONSTRAINT", "CREATE", "CROSS", "CURRENT", "CURRENT_DATE", "CURRENT_TIME",
    "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT", "DEFERRABLE", "DEFERRED", "DELETE",
    "DESC", "DETACH", "DISTINCT", "DO", "DROP", "EACH", "ELSE", "END", "ESCAPE",
    "EXCEPT", "EXCLUDE", "EXCLUSIVE", "EXISTS", "EXPLAIN", "FAIL", "FILTER",
    "FIRST", "FOLLOWING", "FOR", "FOREIGN", "FROM", "FULL", "GENERATED", "GLOB",
    "GROUP", "GROUPS", "HAVING", "IF", "IGNORE", "IMMEDIATE", "IN", "INDEX",
    "INDEXED", "INITIALLY", "INNER", "INSERT", "INSTEAD", "INTERSECT", "INTO",
    "IS", "ISNULL", "JOIN", "KEY", "LAST", "LEFT", "LIKE", "LIMIT", "MATCH",
    "NATURAL", "NO", "NOT", "NOTHING", "NOTNULL", "NULL", "NULLS", "OF", "OFFSET",
    "ON", "OR", "ORDER", "OTHERS", "OUTER", "OVER", "PARTITION", "PLAN", "PRAGMA",
    "PRECEDING", "PRIMARY", "QUERY", "RAISE", "RANGE", "RECURSIVE", "REFERENCES",
    "REGEXP", "REINDEX", "RELEASE", "RENAME", "REPLACE", "RESTRICT", "RIGHT",
    "ROLLBACK", "ROW", "ROWS", "SAVEPOINT", "SELECT", "SET", "TABLE", "TEMP",
    "TEMPORARY", "THEN", "TIES", "TO", "TRANSACTION", "TRIGGER", "UNBOUNDED",
    "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES", "VIEW", "VIRTUAL",
    "WHEN", "WHERE", "WINDOW", "WITH", "WITHOUT"
}

def quote_identifier_if_needed(identifier: str) -> str:
    """Wraps an identifier in double quotes if it's a keyword or needs quoting."""
    if identifier.upper() in SQLITE_KEYWORDS or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        return f'"{identifier}"'
    return identifier


class SQLiteBrowserApp(App):
    """A TUI application to browse and query SQLite databases."""

    CSS = """
    Screen {
        layout: vertical;
    }
    Header {
        dock: top;
    }
    Footer {
        dock: bottom;
    }
    #main-container {
        layout: horizontal;
        height: 1fr;
    }
    #object-list-container {
        width: 40;
        border-right: heavy white;
    }
    #object-tree {
        background: $panel;
        width: 100%;
    }
    #right-pane {
        layout: vertical;
        width: 1fr;
    }
    #sql-status {
        padding: 0 1;
        height: 1;
        background: $primary;
        color: $text;
    }
    #sql-editor-pane {
        padding: 1;
        height: 15;
    }
    TextArea {
        height: 1fr;
        margin-bottom: 1;
    }
    DataTable {
        height: 1fr;
    }
    #sql-results-view {
        border-top: heavy white;
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+r", "reload_objects", "Reload Objects"),
        ("f5", "run_sql", "Execute SQL"),
    ]

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.is_memory_db = (db_path == ":memory:")
        
        if self.is_memory_db:
            self.title = "SQLiteTUI - In-Memory Database"
        else:
            self.title = f"SQLiteTUI - {Path(db_path).name}"
        
        self.sql_history = []

    def compose(self) -> ComposeResult:
        """Compose the layout of the application."""
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="object-list-container"):
                yield Tree("DATABASE", id="object-tree")
            with Vertical(id="right-pane"):
                with TabbedContent(initial="sql-tab"):
                    with TabPane("SQL Editor", id="sql-tab"):
                        with Vertical():
                            with Vertical(id="sql-editor-pane"):
                                yield TextArea(
                                    "SELECT 'Hello World' x, 1+3 y;",
                                    language="sql", id="sql-editor"
                                )
                                yield Button(
                                    "Execute SQL (F5)",
                                    variant="primary", id="run-sql-button"
                                )
                            yield DataTable(id="sql-results-view", zebra_stripes=True)
                    with TabPane("Table Data", id="data-tab"):
                        yield DataTable(id="data-view", zebra_stripes=True)
                    with TabPane("Table DDL", id="ddl-tab"):
                        yield TextArea("", language="sql", id="ddl-view", read_only=True)
                    with TabPane("History / Log", id="history-tab"):
                        yield TextArea("", id="history-view", read_only=True)

        yield Static(id="sql-status")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        self.load_history()
        self.load_db_objects()

    def action_reload_objects(self) -> None:
        """Action to reload the list of database objects."""
        self.load_db_objects()
        self.query_one("#data-view", DataTable).clear(columns=True)
        self.sub_title = "Select an object"

    def load_db_objects(self) -> None:
        """Connects to the DB and populates the object tree."""
        tree = self.query_one("#object-tree", Tree)
        tree.clear()
        tree.root.expand()
        
        tables_node = tree.root.add("TABLES", expand=True)
        views_node = tree.root.add("VIEWS", expand=True)

        try:
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            
            for object_type, parent_node in [("table", tables_node), ("view", views_node)]:
                cur.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='{object_type}' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
                for row in cur.fetchall():
                    object_name = row[0]
                    object_node = parent_node.add(object_name, data={"name": object_name, "type": object_type})
                    cur.execute(f'PRAGMA table_info("{object_name}")')
                    for col in cur.fetchall():
                        name, dtype, notnull, _, pk = col[1:6]
                        tags = []
                        if pk and object_type == "table":
                            tags.append("[bold gold]PK[/]")
                        if notnull:
                            tags.append("[bold red]NN[/]")
                        tag_str = " ".join(tags)
                        label = f"{name} [i]({dtype})[/i] {tag_str}"
                        object_node.add_leaf(label)
        except sqlite3.Error as e:
            self.exit(f"Database error on loading objects: {e}")
        finally:
            if con:
                con.close()

    def load_history(self):
        """Loads query history from the JSON file."""
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r") as f:
                try:
                    self.sql_history = json.load(f)
                except json.JSONDecodeError:
                    self.sql_history = []
        self.update_history_view()

    def save_history(self):
        """Saves query history to the JSON file."""
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.sql_history, f, indent=2)

    def add_to_history(self, sql: str, status: str):
        """Adds a new entry to the history, cleaning the data first."""
        # Clean up the query string to remove trailing semicolons and whitespace
        clean_query = sql.strip().rstrip(";")

        # Clean up the status message to remove Rich console markup
        plain_status = re.sub(r"\[.*?\]", "", status)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": clean_query,
            "status": plain_status,
        }
        self.sql_history.insert(0, entry)
        self.sql_history = self.sql_history[:MAX_HISTORY_ITEMS]
        self.save_history()
        self.update_history_view()

    def update_history_view(self):
        """Updates the history text area."""
        history_text = []
        for entry in self.sql_history:
            ts = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            history_text.append(f"-- {ts} | {entry['status']}")
            history_text.append(entry['query'] + ";")
            history_text.append("")
        self.query_one("#history-view", TextArea).text = "\n".join(history_text)

    def run_sql_query(self, query_script: str) -> None:
        """Runs a SQL script and updates the UI."""
        self.sub_title = "Running Query..."
        results_table = self.query_one("#sql-results-view", DataTable)
        results_table.clear(columns=True)
        status_bar = self.query_one("#sql-status", Static)
        status_message = ""

        con = None
        try:
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            statements = [s.strip() for s in query_script.split(";") if s.strip()]
            if not statements:
                status_message = "[bold yellow]No SQL statement to execute.[/]"
                status_bar.update(status_message)
                self.add_to_history(query_script, "No statement")
                return

            for statement in statements[:-1]:
                cur.execute(statement)

            last_statement = statements[-1]
            cur.execute(last_statement)

            if last_statement.upper().startswith("SELECT"):
                rows = cur.fetchall()
                if not rows:
                    status_message = "[bold yellow]SELECT returned 0 rows.[/]"
                else:
                    columns = list(rows[0].keys())
                    results_table.add_columns(*columns)
                    results_table.add_rows([tuple(row) for row in rows])
                    status_message = f"[bold green]SELECT returned {len(rows)} rows.[/]"
            else:
                status_message = f"[bold green]{cur.rowcount} rows affected by the last statement.[/]"
            
            con.commit()
            self.load_db_objects() # Refresh schema in case of DDL changes

        except sqlite3.Error as e:
            if con:
                con.rollback()
            status_message = f"[bold red]SQL Error: {e}[/]"
            self.sub_title = "Query Error"
        finally:
            if con:
                con.close()
            
            status_bar.update(status_message)
            self.sub_title = "Query Complete"
            self.add_to_history(query_script, status_message)

    @on(Tree.NodeSelected, "#object-tree")
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Called when an object is selected, populating the data and DDL tabs."""
        node = event.node
        if not node.data: # It's a parent node like "TABLES" or a column
            # if we click on a column, still treat it as clicking the parent table/view
            if node.parent and node.parent.data:
                node = node.parent
            else:
                return

        object_name = node.data["name"]
        object_type = node.data["type"]
        
        tabs = self.query_one(TabbedContent)
        data_view_table = self.query_one("#data-view", DataTable)
        data_view_table.clear(columns=True)
        ddl_view = self.query_one("#ddl-view", TextArea)
        status_bar = self.query_one("#sql-status", Static)

        con = None
        try:
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            cur.execute(f'SELECT * FROM {quote_identifier_if_needed(object_name)} LIMIT 1000')
            rows = cur.fetchall()
            if rows:
                columns = list(rows[0].keys())
                data_view_table.add_columns(*columns)
                data_view_table.add_rows([tuple(row) for row in rows])
            status_bar.update(f"Viewing {object_type} '{object_name}'. {len(rows)} rows.")

            ddl_parts = []
            cur.execute("SELECT sql FROM sqlite_master WHERE type=? AND name=?", (object_type, object_name))
            result = cur.fetchone()
            if result: ddl_parts.append(result[0] + ";")

            if object_type == "table":
                # Logic for table-specific DDL like indexes and triggers
                cur.execute(f'PRAGMA index_list("{object_name}")')
                pk_index_name = next((idx[1] for idx in cur.fetchall() if idx[3] == 'pk'), None)

                if pk_index_name:
                    cur.execute(f'PRAGMA index_info("{pk_index_name}")')
                    cols = [row[2] for row in cur.fetchall()]
                    if cols:
                        q_idx = quote_identifier_if_needed(pk_index_name)
                        q_tbl = quote_identifier_if_needed(object_name)
                        col_str = ", ".join(quote_identifier_if_needed(c) for c in cols)
                        ddl_parts.append(f'\n-- Primary Key Index\nCREATE UNIQUE INDEX {q_idx} ON {q_tbl}({col_str});')

                cur.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (object_name,))
                indexes = [idx[0] + ";" for idx in cur.fetchall() if pk_index_name is None or pk_index_name not in idx[0]]
                if indexes:
                    ddl_parts.append("\n-- Other Indexes")
                    ddl_parts.extend(indexes)

                cur.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (object_name,))
                triggers = [trg[0] + ";" for trg in cur.fetchall() if trg[0]]
                if triggers:
                    ddl_parts.append("\n-- Triggers")
                    ddl_parts.extend(triggers)

            ddl_view.text = "\n".join(ddl_parts)
            tabs.active = "data-tab"

        except sqlite3.Error as e:
            status_bar.update(f"[bold red]Error loading object: {e}[/]")
        finally:
            if con:
                con.close()

    def action_run_sql(self) -> None:
        sql_query = self.query_one("#sql-editor", TextArea).text
        self.run_sql_query(sql_query)

    @on(Button.Pressed, "#run-sql-button")
    def on_run_sql_button_pressed(self) -> None:
        self.action_run_sql()


if __name__ == "__main__":
    # Default to in-memory database if no argument provided
    if len(sys.argv) == 1:
        print("No database file specified. Using in-memory database (:memory:)")
        db_path = ":memory:"
    else:
        db_filename = sys.argv[1]
        db_path_obj = Path(db_filename)
        
        # If file doesn't exist, inform user we're creating a new database
        if not db_path_obj.is_file():
            print(f"Database file '{db_filename}' does not exist. Creating new database...")
        
        db_path = str(db_path_obj)

    app = SQLiteBrowserApp(db_path=db_path)
    app.run()
