import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
import io
import contextlib
import builtins
import re
import ast

st.set_page_config(page_title="AI DataLab", page_icon="💻", layout="wide")

st.title("💻 AI DataLab")
st.caption("Browser-Based Python & SQL Practice Lab")


# ============================================================
# Utility
# ============================================================

def clean_name(filename):
    name = re.sub(r"\W+", "_", filename.rsplit(".", 1)[0])
    name = re.sub(r"_+", "_", name).strip("_") or "dataset"

    if name[0].isdigit():
        name = "dataset_" + name

    return name


def make_env(dataframes):
    original_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed = {
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn"
        }

        if name.split(".")[0] not in allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed in this training lab."
            )

        return original_import(
            name, globals, locals, fromlist, level
        )

    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "enumerate": enumerate,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "__import__": safe_import
    }

    env = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns
    }

    for name, dataframe in dataframes.items():
        env[name] = dataframe

    if dataframes:
        env["df"] = next(iter(dataframes.values()))

    return env


def execute_cell(code, env):
    """
    Execute a cell and display:
    - print() output
    - last expression output, e.g. df.head()
    - DataFrames
    - Series
    - charts
    """

    output = io.StringIO()
    result = None

    plt.close("all")

    try:
        tree = ast.parse(code)

        # If the last statement is an expression such as:
        # df.head()
        # df.describe()
        # df["amount"].sum()
        # evaluate it so the result can be displayed.
        body = tree.body

        if body and isinstance(body[-1], ast.Expr):

            last_expression = body[-1].value

            body_without_last = body[:-1]

            if body_without_last:
                module_without_last = ast.Module(
                    body=body_without_last,
                    type_ignores=[]
                )

                ast.fix_missing_locations(
                    module_without_last
                )

                with contextlib.redirect_stdout(output):
                    exec(
                        compile(
                            module_without_last,
                            "<cell>",
                            "exec"
                        ),
                        env,
                        env
                    )

            expression = ast.Expression(
                body=last_expression
            )

            ast.fix_missing_locations(expression)

            with contextlib.redirect_stdout(output):
                result = eval(
                    compile(
                        expression,
                        "<cell>",
                        "eval"
                    ),
                    env,
                    env
                )

        else:

            with contextlib.redirect_stdout(output):
                exec(code, env, env)

        # Get chart
        fig = None

        if plt.get_fignums():
            fig = plt.gcf()

        return (
            True,
            output.getvalue(),
            result,
            fig,
            None
        )

    except Exception as e:

        plt.close("all")

        return (
            False,
            output.getvalue(),
            None,
            None,
            f"{type(e).__name__}: {e}"
        )


def extract_notebook_cells(data):
    notebook = json.loads(
        data.decode("utf-8")
    )

    cells = []

    for cell in notebook.get("cells", []):

        if cell.get("cell_type") == "code":

            source = cell.get("source", [])

            if isinstance(source, list):
                source = "".join(source)

            cells.append(source)

    return cells


def show_result(result):
    """
    Display normal Python/Jupyter-style results.
    """

    if result is None:
        return

    if isinstance(result, pd.DataFrame):

        st.dataframe(
            result,
            use_container_width=True
        )

    elif isinstance(result, pd.Series):

        st.dataframe(
            result.to_frame(),
            use_container_width=True
        )

    elif isinstance(result, np.ndarray):

        st.write(result)

    else:

        st.write(result)


# ============================================================
# Session State
# ============================================================

if "nb_cells" not in st.session_state:
    st.session_state.nb_cells = []

if "nb_outputs" not in st.session_state:
    st.session_state.nb_outputs = {}

if "nb_env" not in st.session_state:
    st.session_state.nb_env = None

if "notebook_id" not in st.session_state:
    st.session_state.notebook_id = None


# ============================================================
# Navigation
# ============================================================

lab = st.sidebar.radio(
    "Select Lab",
    [
        "Home",
        "Python Lab",
        "SQL Lab"
    ]
)


# ============================================================
# HOME
# ============================================================

if lab == "Home":

    st.header("Welcome to AI DataLab")

    st.markdown("""
### Participant Practice Lab

Participants can:

- Upload multiple CSV datasets
- Run Python code in the browser
- Upload `.ipynb` notebooks
- Edit notebook cells
- Add new cells
- Delete cells
- Run cells one by one
- Run all cells
- See `print()` output
- See `df.head()` / `df.describe()` output
- Create Matplotlib / Seaborn charts
- Practice SQL
- Run SQL joins
- Download SQL results

### Example

Upload:

`airbnb.csv`

Then in a notebook cell:

```python
airbnb.head()
```

The result will be displayed automatically.
""")


# ============================================================
# PYTHON LAB
# ============================================================

elif lab == "Python Lab":

    st.header("🐍 Python Practice Lab")

    uploaded_files = st.file_uploader(
        "Step 1 — Upload one or more CSV datasets",
        type=["csv"],
        accept_multiple_files=True,
        key="python_csv_upload"
    )

    if uploaded_files:

        dataframes = {}

        for uploaded_file in uploaded_files:

            try:

                dataframe = pd.read_csv(
                    uploaded_file
                )

                variable_name = clean_name(
                    uploaded_file.name
                )

                original_name = variable_name
                counter = 2

                while variable_name in dataframes:

                    variable_name = (
                        f"{original_name}_{counter}"
                    )

                    counter += 1

                dataframes[
                    variable_name
                ] = dataframe

            except Exception as e:

                st.error(
                    f"Could not read "
                    f"{uploaded_file.name}: "
                    f"{type(e).__name__}: {e}"
                )

        if dataframes:

            st.success(
                f"{len(dataframes)} dataset(s) "
                f"loaded successfully."
            )

            st.subheader(
                "Uploaded Datasets"
            )

            for name, dataframe in dataframes.items():

                with st.expander(
                    f"📄 {name} — "
                    f"{len(dataframe):,} rows × "
                    f"{len(dataframe.columns):,} columns"
                ):

                    st.dataframe(
                        dataframe.head(10),
                        use_container_width=True
                    )

                    st.write(
                        "Columns:",
                        list(dataframe.columns)
                    )

            st.info(
                "Dataset variable names are based on "
                "the CSV filenames. The first dataset "
                "is also available as `df`."
            )

            # ------------------------------------------------
            # Direct Python
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "Option 1 — Write Python Code"
            )

            names = list(dataframes.keys())

            example = (
                "# Example\n"
                f"{names[0]}.head()\n"
            )

            code = st.text_area(
                "Python Code",
                value=example,
                height=250,
                key="direct_python"
            )

            if st.button(
                "▶ Run Python Code",
                type="primary"
            ):

                (
                    ok,
                    output,
                    result,
                    fig,
                    error
                ) = execute_cell(
                    code,
                    make_env(dataframes)
                )

                if ok:

                    st.success(
                        "Code executed successfully."
                    )

                    if output:

                        st.subheader("Output")

                        st.code(
                            output,
                            language="text"
                        )

                    if result is not None:

                        st.subheader(
                            "Result"
                        )

                        show_result(result)

                    if fig is not None:

                        st.subheader(
                            "Chart Output"
                        )

                        st.pyplot(fig)

                        plt.close(fig)

                else:

                    st.error(error)

            # ------------------------------------------------
            # Notebook
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "Option 2 — Upload & Edit Jupyter Notebook"
            )

            nb_file = st.file_uploader(
                "Upload .ipynb Notebook",
                type=["ipynb"],
                key="notebook_upload"
            )

            if nb_file:

                try:

                    cells_from_file = (
                        extract_notebook_cells(
                            nb_file.getvalue()
                        )
                    )

                    if not cells_from_file:

                        st.warning(
                            "No Python code cells "
                            "were found."
                        )

                    else:

                        notebook_id = (
                            nb_file.name,
                            len(nb_file.getvalue()),
                            tuple(dataframes.keys())
                        )

                        if (
                            st.session_state.notebook_id
                            != notebook_id
                        ):

                            st.session_state.notebook_id = (
                                notebook_id
                            )

                            st.session_state.nb_cells = (
                                cells_from_file.copy()
                            )

                            st.session_state.nb_env = (
                                make_env(dataframes)
                            )

                            st.session_state.nb_outputs = {}

                        st.success(
                            f"Notebook loaded — "
                            f"{len(st.session_state.nb_cells)} "
                            f"code cells."
                        )

                        # Toolbar

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            if st.button(
                                "➕ Add New Cell",
                                key="add_cell_top",
                                use_container_width=True
                            ):

                                st.session_state.nb_cells.append(
                                    "# Write Python code here"
                                )

                                st.rerun()

                        with col2:

                            if st.button(
                                "▶ Run All Cells",
                                type="primary",
                                key="run_all",
                                use_container_width=True
                            ):

                                st.session_state.nb_env = (
                                    make_env(dataframes)
                                )

                                st.session_state.nb_outputs = {}

                                for i, cell in enumerate(
                                    st.session_state.nb_cells
                                ):

                                    (
                                        ok,
                                        output,
                                        result,
                                        fig,
                                        error
                                    ) = execute_cell(
                                        cell,
                                        st.session_state.nb_env
                                    )

                                    if ok:

                                        st.session_state.nb_outputs[
                                            i
                                        ] = {
                                            "output": output,
                                            "result": result,
                                            "fig": fig
                                        }

                                    else:

                                        st.session_state.nb_outputs[
                                            i
                                        ] = {
                                            "error": error,
                                            "output": output
                                        }

                                st.rerun()

                        with col3:

                            if st.button(
                                "🔄 Reset Notebook",
                                key="reset_nb",
                                use_container_width=True
                            ):

                                st.session_state.nb_cells = (
                                    cells_from_file.copy()
                                )

                                st.session_state.nb_env = (
                                    make_env(dataframes)
                                )

                                st.session_state.nb_outputs = {}

                                st.rerun()

                        st.divider()

                        # Editable notebook cells

                        for i in range(
                            len(
                                st.session_state.nb_cells
                            )
                        ):

                            st.markdown(
                                f"### Cell {i + 1}"
                            )

                            edited_code = st.text_area(
                                f"Edit Cell {i + 1}",
                                value=st.session_state.nb_cells[i],
                                height=180,
                                key=f"cell_editor_{i}"
                            )

                            st.session_state.nb_cells[i] = (
                                edited_code
                            )

                            c1, c2 = st.columns(2)

                            with c1:

                                if st.button(
                                    f"▶ Run Cell {i + 1}",
                                    key=f"run_cell_{i}",
                                    use_container_width=True
                                ):

                                    (
                                        ok,
                                        output,
                                        result,
                                        fig,
                                        error
                                    ) = execute_cell(
                                        st.session_state.nb_cells[i],
                                        st.session_state.nb_env
                                    )

                                    if ok:

                                        st.session_state.nb_outputs[
                                            i
                                        ] = {
                                            "output": output,
                                            "result": result,
                                            "fig": fig
                                        }

                                    else:

                                        st.session_state.nb_outputs[
                                            i
                                        ] = {
                                            "error": error,
                                            "output": output
                                        }

                                    st.rerun()

                            with c2:

                                if st.button(
                                    f"🗑 Delete Cell {i + 1}",
                                    key=f"delete_cell_{i}",
                                    use_container_width=True
                                ):

                                    st.session_state.nb_cells.pop(
                                        i
                                    )

                                    new_outputs = {}

                                    for key, value in (
                                        st.session_state.nb_outputs.items()
                                    ):

                                        if key < i:

                                            new_outputs[key] = value

                                        elif key > i:

                                            new_outputs[
                                                key - 1
                                            ] = value

                                    st.session_state.nb_outputs = (
                                        new_outputs
                                    )

                                    st.rerun()

                            result_data = (
                                st.session_state.nb_outputs.get(
                                    i
                                )
                            )

                            if result_data:

                                if "error" in result_data:

                                    st.error(
                                        result_data["error"]
                                    )

                                    if result_data.get(
                                        "output"
                                    ):

                                        st.code(
                                            result_data[
                                                "output"
                                            ],
                                            language="text"
                                        )

                                else:

                                    if result_data.get(
                                        "output"
                                    ):

                                        st.markdown(
                                            "**Output**"
                                        )

                                        st.code(
                                            result_data[
                                                "output"
                                            ],
                                            language="text"
                                        )

                                    if result_data.get(
                                        "result"
                                    ) is not None:

                                        st.markdown(
                                            "**Result**"
                                        )

                                        show_result(
                                            result_data[
                                                "result"
                                            ]
                                        )

                                    if result_data.get(
                                        "fig"
                                    ) is not None:

                                        st.markdown(
                                            "**Chart Output**"
                                        )

                                        st.pyplot(
                                            result_data[
                                                "fig"
                                            ]
                                        )

                            st.divider()

                        if st.button(
                            "➕ Add New Cell",
                            key="add_cell_bottom",
                            use_container_width=True
                        ):

                            st.session_state.nb_cells.append(
                                "# Write Python code here"
                            )

                            st.rerun()

                        st.info(
                            "Notebook cells behave like Jupyter "
                            "cells: edit them, run them individually, "
                            "and use variables created in earlier cells."
                        )

                except Exception as e:

                    st.error(
                        f"Could not read notebook: "
                        f"{type(e).__name__}: {e}"
                    )


# ============================================================
# SQL LAB
# ============================================================

elif lab == "SQL Lab":

    st.header("🗄️ SQL Practice Lab")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV datasets",
        type=["csv"],
        accept_multiple_files=True,
        key="sql_csv_upload"
    )

    if uploaded_files:

        sql_dataframes = {}

        for uploaded_file in uploaded_files:

            try:

                dataframe = pd.read_csv(
                    uploaded_file
                )

                table_name = clean_name(
                    uploaded_file.name
                )

                original_name = table_name
                counter = 2

                while table_name in sql_dataframes:

                    table_name = (
                        f"{original_name}_{counter}"
                    )

                    counter += 1

                sql_dataframes[
                    table_name
                ] = dataframe

            except Exception as e:

                st.error(
                    f"Could not read "
                    f"{uploaded_file.name}: "
                    f"{type(e).__name__}: {e}"
                )

        if sql_dataframes:

            st.success(
                f"{len(sql_dataframes)} "
                f"SQL table(s) loaded."
            )

            st.subheader(
                "Available SQL Tables"
            )

            for table_name, dataframe in (
                sql_dataframes.items()
            ):

                st.write(
                    f"**{table_name}** — "
                    f"{len(dataframe):,} rows × "
                    f"{len(dataframe.columns):,} columns"
                )

            first_table = next(
                iter(sql_dataframes)
            )

            st.divider()

            query = st.text_area(
                "SQL Query",
                value=(
                    f"SELECT * FROM "
                    f"{first_table} LIMIT 10;"
                ),
                height=220
            )

            if st.button(
                "▶ Run SQL",
                type="primary"
            ):

                try:

                    conn = sqlite3.connect(
                        ":memory:"
                    )

                    for table_name, dataframe in (
                        sql_dataframes.items()
                    ):

                        dataframe.to_sql(
                            table_name,
                            conn,
                            index=False,
                            if_exists="replace"
                        )

                    result = pd.read_sql_query(
                        query,
                        conn
                    )

                    conn.close()

                    st.success(
                        f"Query executed successfully. "
                        f"{len(result):,} rows returned."
                    )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇ Download Result as CSV",
                        data=result.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="sql_result.csv",
                        mime="text/csv"
                    )

                except Exception as e:

                    st.error(
                        f"{type(e).__name__}: {e}"
                    )
