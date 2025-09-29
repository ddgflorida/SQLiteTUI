# SQLiteTUI - Browse SQLite databases
import sys
import sqlite3
import re
import json
import csv
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

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
    Input,
)
from textual import on

# --- History Configuration ---
HISTORY_DIR = Path.home() / ".sqlite_browser_history"
MAX_HISTORY_ITEMS = 100

# Ensure history directory exists
HISTORY_DIR.mkdir(exist_ok=True)

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

def quote_identifier(identifier: str) -> str:
    """Wraps an identifier in double quotes, escaping any internal quotes."""
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'

def quote_identifier_if_needed(identifier: str) -> str:
    """Wraps an identifier in double quotes if it's a keyword or needs quoting."""
    if identifier.upper() in SQLITE_KEYWORDS or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        return quote_identifier(identifier)
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
    #search-container {
        height: 3;
        padding: 0 1;
    }
    #object-tree {
        background: $panel;
        width: 100%;
        height: 1fr;
    }
    #right-pane {
        layout: vertical;
        width: 1fr;
    }
    #sql-status {
        padding: 0 1;
        height: 2;
        background: $primary;
        color: $text;
        dock: top;
    }
    #sql-editor-pane {
        padding: 1;
        height: 15;
        min-height: 10;
    }
    #sql-editor-pane.expanded {
        height: 30;
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
        ("ctrl+e", "toggle_editor_size", "Expand/Collapse Editor"),
        ("enter", "load_from_history", "Load Query from History"),
        ("ctrl+s", "export_results", "Export Results"),
        ("ctrl+d", "clear_history", "Clear History"),
    ]

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.is_memory_db = (db_path == ":memory:")
        
        if self.is_memory_db:
            self.title = "SQLiteTUI - In-Memory Database"
            # Use a generic history file for in-memory databases
            self.history_file = HISTORY_DIR / "memory_db_history.json"
        else:
            self.title = f"SQLiteTUI - {Path(db_path).name}"
            # Use full resolved path for database-specific history file
            full_path = Path(db_path).resolve()
            # Replace path separators with underscores and sanitize
            safe_path = str(full_path).replace('/', '_').replace('\\', '_').replace(':', '_')
            safe_path = re.sub(r'[^\w\-._]', '_', safe_path)
            self.history_file = HISTORY_DIR / f"{safe_path}_history.json"
        
        self.sql_history = []
        self._connection = None
        self._all_objects = []  # Store all objects for filtering
        self._editor_expanded = False
        self._last_query_results = []  # Store last query results for export
        self._last_query_columns = []  # Store column names

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if self.is_memory_db:
            # For in-memory databases, maintain a persistent connection
            if self._connection is None:
                self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._connection.row_factory = sqlite3.Row
            yield self._connection
        else:
            # For file databases, create new connections each time
            con = sqlite3.connect(self.db_path, check_same_thread=False)
            con.row_factory = sqlite3.Row
            try:
                yield con
            finally:
                con.close()

    def compose(self) -> ComposeResult:
        """Compose the layout of the application."""
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="object-list-container"):
                with Vertical(id="search-container"):
                    yield Input(placeholder="Search objects...", id="object-search")
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

    def on_unmount(self) -> None:
        """Clean up resources when app closes."""
        if self._connection:
            self._connection.close()

    def action_reload_objects(self) -> None:
        """Action to reload the list of database objects."""
        self.load_db_objects()
        self.query_one("#data-view", DataTable).clear(columns=True)
        self.query_one("#object-search", Input).value = ""
        self.sub_title = "Select an object"

    def load_db_objects(self, filter_text: str = "") -> None:
        """Connects to the DB and populates the object tree."""
        tree = self.query_one("#object-tree", Tree)
        tree.clear()
        tree.root.expand()
        
        tables_node = tree.root.add("TABLES", expand=True)
        views_node = tree.root.add("VIEWS", expand=True)
        
        status_bar = self.query_one("#sql-status", Static)

        try:
            with self.get_connection() as con:
                cur = con.cursor()
                
                # Store all objects if not filtering
                if not filter_text:
                    self._all_objects = []
                
                for object_type, parent_node in [("table", tables_node), ("view", views_node)]:
                    # Use parameterized query to prevent SQL injection
                    cur.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type=? AND name NOT LIKE 'sqlite_%'
                        ORDER BY name
                    """, (object_type,))
                    
                    for row in cur.fetchall():
                        object_name = row[0]
                        
                        # Apply filter
                        if filter_text and filter_text.lower() not in object_name.lower():
                            continue
                        
                        if not filter_text:
                            self._all_objects.append((object_name, object_type))
                        
                        object_node = parent_node.add(
                            object_name, 
                            data={"name": object_name, "type": object_type}
                        )
                        
                        # Use quote_identifier for PRAGMA
                        cur.execute(f'PRAGMA table_info({quote_identifier(object_name)})')
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
                
                status_bar.update("Objects loaded successfully")
                
        except sqlite3.Error as e:
            status_bar.update(f"[bold red]Error loading objects: {e}[/]")

    @on(Input.Changed, "#object-search")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Filter objects based on search input."""
        self.load_db_objects(filter_text=event.value)

    def load_history(self):
        """Loads query history from the JSON file."""
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                try:
                    self.sql_history = json.load(f)
                except json.JSONDecodeError:
                    self.sql_history = []
        self.update_history_view()

    def save_history(self):
        """Saves query history to the JSON file."""
        with open(self.history_file, "w") as f:
            json.dump(self.sql_history, f, indent=2)

    def add_to_history(self, sql: str, status: str, success: bool):
        """Adds a new entry to the history, cleaning the data first."""
        # Clean up the query string to remove trailing semicolons and whitespace
        clean_query = sql.strip().rstrip(";")

        # Clean up the status message to remove Rich console markup
        plain_status = re.sub(r"\[.*?\]", "", status)

        # Check if this is the same as the most recent query
        if self.sql_history and self.sql_history[0]["query"] == clean_query:
            # Update the timestamp and status of the most recent entry instead
            self.sql_history[0]["timestamp"] = datetime.now().isoformat()
            self.sql_history[0]["status"] = plain_status
            self.sql_history[0]["success"] = success
        else:
            # Add new entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": clean_query,
                "status": plain_status,
                "success": success,
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
            success_marker = "✓" if entry.get('success', True) else "✗"
            history_text.append(f"-- {ts} | {success_marker} | {entry['status']}")
            history_text.append(entry['query'] + ";")
            history_text.append("")
        history_view = self.query_one("#history-view", TextArea)
        history_view.text = "\n".join(history_text)

    def run_sql_query(self, query_script: str) -> None:
        """Runs a SQL script and updates the UI."""
        self.sub_title = "Running Query..."
        results_table = self.query_one("#sql-results-view", DataTable)
        results_table.clear(columns=True)
        status_bar = self.query_one("#sql-status", Static)
        status_message = ""
        success = False

        try:
            with self.get_connection() as con:
                cur = con.cursor()

                statements = [s.strip() for s in query_script.split(";") if s.strip()]
                if not statements:
                    status_message = "[bold yellow]No SQL statement to execute.[/]"
                    status_bar.update(status_message)
                    self.add_to_history(query_script, "No statement", False)
                    return

                # Execute all statements except the last
                for statement in statements[:-1]:
                    cur.execute(statement)

                # Execute the last statement
                last_statement = statements[-1]
                cur.execute(last_statement)

                # Check if it's a SELECT statement
                if last_statement.strip().upper().startswith("SELECT") or \
                   last_statement.strip().upper().startswith("WITH"):
                    rows = cur.fetchall()
                    if not rows:
                        status_message = "[bold yellow]Query returned 0 rows.[/]"
                        self._last_query_results = []
                        self._last_query_columns = []
                    else:
                        columns = list(rows[0].keys())
                        results_table.add_columns(*columns)
                        results_table.add_rows([tuple(row) for row in rows])
                        status_message = f"[bold yellow]Query returned {len(rows)} rows.[/]"
                        # Store results for export
                        self._last_query_results = [tuple(row) for row in rows]
                        self._last_query_columns = columns
                else:
                    status_message = f"[bold yellow]{cur.rowcount} rows affected.[/]"
                    self._last_query_results = []
                    self._last_query_columns = []
                
                con.commit()
                success = True
                self.load_db_objects() # Refresh schema in case of DDL changes

        except sqlite3.Error as e:
            status_message = f"[bold red]SQL Error: {e}[/]"
            self.sub_title = "Query Error"
        finally:
            status_bar.update(status_message)
            self.sub_title = "Query Complete"
            self.add_to_history(query_script, status_message, success)

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

        try:
            with self.get_connection() as con:
                cur = con.cursor()

                # Use proper identifier quoting
                cur.execute(f'SELECT * FROM {quote_identifier(object_name)} LIMIT 1000')
                rows = cur.fetchall()
                if rows:
                    columns = list(rows[0].keys())
                    data_view_table.add_columns(*columns)
                    data_view_table.add_rows([tuple(row) for row in rows])
                
                row_count = len(rows)
                limit_msg = " (limited to 1000)" if row_count == 1000 else ""
                status_bar.update(f"Viewing {object_type} '{object_name}'. {row_count} rows{limit_msg}.")

                ddl_parts = []
                cur.execute(
                    "SELECT sql FROM sqlite_master WHERE type=? AND name=?", 
                    (object_type, object_name)
                )
                result = cur.fetchone()
                if result: 
                    ddl_parts.append(result[0] + ";")

                if object_type == "table":
                    # Get indexes
                    cur.execute(f'PRAGMA index_list({quote_identifier(object_name)})')
                    indexes_info = cur.fetchall()
                    pk_index_name = next((idx[1] for idx in indexes_info if idx[3] == 'pk'), None)

                    if pk_index_name:
                        cur.execute(f'PRAGMA index_info({quote_identifier(pk_index_name)})')
                        cols = [row[2] for row in cur.fetchall()]
                        if cols:
                            q_idx = quote_identifier(pk_index_name)
                            q_tbl = quote_identifier(object_name)
                            col_str = ", ".join(quote_identifier(c) for c in cols)
                            ddl_parts.append(f'\n-- Primary Key Index\nCREATE UNIQUE INDEX {q_idx} ON {q_tbl}({col_str});')

                    cur.execute(
                        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", 
                        (object_name,)
                    )
                    indexes = [idx[0] + ";" for idx in cur.fetchall() 
                              if pk_index_name is None or pk_index_name not in idx[0]]
                    if indexes:
                        ddl_parts.append("\n-- Other Indexes")
                        ddl_parts.extend(indexes)

                    cur.execute(
                        "SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", 
                        (object_name,)
                    )
                    triggers = [trg[0] + ";" for trg in cur.fetchall() if trg[0]]
                    if triggers:
                        ddl_parts.append("\n-- Triggers")
                        ddl_parts.extend(triggers)

                ddl_view.text = "\n".join(ddl_parts)
                tabs.active = "data-tab"

        except sqlite3.Error as e:
            status_bar.update(f"[bold red]Error loading object: {e}[/]")

    def action_run_sql(self) -> None:
        sql_query = self.query_one("#sql-editor", TextArea).text
        self.run_sql_query(sql_query)

    def action_toggle_editor_size(self) -> None:
        """Toggle the SQL editor between normal and expanded size."""
        editor_pane = self.query_one("#sql-editor-pane")
        self._editor_expanded = not self._editor_expanded
        if self._editor_expanded:
            editor_pane.add_class("expanded")
        else:
            editor_pane.remove_class("expanded")

    def action_load_from_history(self) -> None:
        """Load query from history when Enter is pressed (only in history tab)."""
        # Only work if we're on the history tab
        tabs = self.query_one(TabbedContent)
        if tabs.active != "history-tab":
            return
        
        # Check if history view has focus
        if not self.query_one("#history-view", TextArea).has_focus:
            return
        
        self._load_query_from_history()

    def action_export_results(self) -> None:
        """Export the last query results to CSV and JSON."""
        status_bar = self.query_one("#sql-status", Static)
        
        if not self._last_query_results or not self._last_query_columns:
            status_bar.update("[bold yellow]No query results to export. Run a SELECT query first.[/]")
            self.notify("No query results to export", severity="warning", timeout=3)
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to CSV
        csv_filename = f"export_{timestamp}.csv"
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self._last_query_columns)
                writer.writerows(self._last_query_results)
            csv_success = True
            csv_error = None
        except Exception as e:
            csv_success = False
            csv_error = str(e)
        
        # Export to JSON
        json_filename = f"export_{timestamp}.json"
        try:
            json_data = [
                dict(zip(self._last_query_columns, row))
                for row in self._last_query_results
            ]
            with open(json_filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(json_data, jsonfile, indent=2, default=str)
            json_success = True
            json_error = None
        except Exception as e:
            json_success = False
            json_error = str(e)
        
        # Update status and show notification
        messages = []
        if csv_success:
            messages.append(f"CSV: {csv_filename}")
        else:
            messages.append(f"CSV failed: {csv_error}")
        
        if json_success:
            messages.append(f"JSON: {json_filename}")
        else:
            messages.append(f"JSON failed: {json_error}")
        
        if csv_success and json_success:
            status_msg = f"[bold yellow]Exported {len(self._last_query_results)} rows → {' | '.join(messages)}[/]"
            status_bar.update(status_msg)
            self.notify(
                f"✓ Exported {len(self._last_query_results)} rows to {csv_filename} and {json_filename}",
                severity="information",
                timeout=5
            )
        elif csv_success or json_success:
            status_msg = f"[bold magenta]Partial export: {' | '.join(messages)}[/]"
            status_bar.update(status_msg)
            self.notify(f"⚠ Partial export: {' | '.join(messages)}", severity="warning", timeout=5)
        else:
            status_msg = f"[bold red]Export failed: {' | '.join(messages)}[/]"
            status_bar.update(status_msg)
            self.notify(f"✗ Export failed", severity="error", timeout=5)

    def action_clear_history(self) -> None:
        """Clear the query history for this database."""
        if not self.sql_history:
            status_bar = self.query_one("#sql-status", Static)
            status_bar.update("[bold yellow]History is already empty.[/]")
            self.notify("History is already empty", severity="information", timeout=3)
            return
        
        # Clear the history
        history_count = len(self.sql_history)
        self.sql_history = []
        self.save_history()
        self.update_history_view()
        
        # Update status and notify
        status_bar = self.query_one("#sql-status", Static)
        status_bar.update(f"[bold yellow]Cleared {history_count} history entries.[/]")
        self.notify(f"✓ Cleared {history_count} history entries", severity="information", timeout=3)

    @on(Button.Pressed, "#run-sql-button")
    def on_run_sql_button_pressed(self) -> None:
        self.action_run_sql()

    def _load_query_from_history(self) -> None:
        """Load selected query from history into the SQL editor."""
        history_view = self.query_one("#history-view", TextArea)
        
        # Get the currently selected line
        cursor_row = history_view.cursor_location[0]
        lines = history_view.text.split("\n")
        
        if cursor_row >= len(lines):
            return
        
        # Find the query associated with this line
        # Queries are on lines that don't start with "--" and aren't empty
        query_line = None
        
        # If we're on a comment line, get the next line
        if lines[cursor_row].startswith("--"):
            if cursor_row + 1 < len(lines) and lines[cursor_row + 1].strip():
                query_line = lines[cursor_row + 1]
        elif lines[cursor_row].strip():
            query_line = lines[cursor_row]
        
        # Load the query into the editor if found
        if query_line:
            sql_editor = self.query_one("#sql-editor", TextArea)
            # Remove trailing semicolon for cleaner editing
            clean_query = query_line.rstrip(";").strip()
            if clean_query:
                sql_editor.text = clean_query
                # Switch to SQL Editor tab
                tabs = self.query_one(TabbedContent)
                tabs.active = "sql-tab"


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
