
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3, json, io, contextlib, builtins

st.set_page_config(page_title="AI DataLab", page_icon="💻", layout="wide")
st.title("💻 AI DataLab")
st.caption("Browser-Based Python & SQL Practice Lab")

def run_code(code, df):
    out = io.StringIO()
    original_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in {"pandas","numpy","matplotlib","seaborn"}:
            raise ImportError(f"Import of '{name}' is not allowed in this training lab.")
        return original_import(name, globals, locals, fromlist, level)

    safe_builtins = {
        "print": print, "len": len, "range": range, "sum": sum,
        "min": min, "max": max, "abs": abs, "round": round,
        "sorted": sorted, "enumerate": enumerate, "list": list,
        "dict": dict, "set": set, "tuple": tuple,
        "__import__": safe_import
    }
    env = {"__builtins__": safe_builtins, "pd": pd, "np": np,
           "plt": plt, "sns": sns, "df": df}
    plt.close("all")
    try:
        with contextlib.redirect_stdout(out):
            exec(code, env, env)
        fig = plt.gcf() if plt.gcf().get_axes() else None
        return True, out.getvalue(), fig, None
    except Exception as e:
        plt.close("all")
        return False, out.getvalue(), None, f"{type(e).__name__}: {e}"

def notebook_code(data):
    nb = json.loads(data.decode("utf-8"))
    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            cells.append("".join(src) if isinstance(src, list) else src)
    return cells

lab = st.sidebar.radio("Select Lab", ["Home", "Python Lab", "SQL Lab"])

if lab == "Home":
    st.header("Welcome to AI DataLab")
    st.write("Upload data, run Python or Jupyter notebooks, create charts, and practice SQL.")
    st.write("Day 1: Python + Pandas + Visualization | Day 2: SQL + Data Analytics")

elif lab == "Python Lab":
    st.header("🐍 Python Practice Lab")
    csv_file = st.file_uploader("Step 1: Upload CSV dataset", type=["csv"], key="csv")
    if csv_file:
        df = pd.read_csv(csv_file)
        st.success(f"Loaded: {csv_file.name}")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Option 1 — Write Python Code")
        example = '# Example\nprint(df.head())\n\nimport matplotlib.pyplot as plt\nplt.hist(df["#Passengers"])\nplt.xlabel("Passengers")\nplt.ylabel("Frequency")\nplt.title("Passenger Distribution")\nplt.show()'
        code = st.text_area("Python", value=example, height=280)

        if st.button("▶ Run Python Code", type="primary"):
            ok, output, fig, err = run_code(code, df)
            if ok:
                st.success("Code executed successfully.")
                if output: st.code(output)
                if fig is not None:
                    st.subheader("Chart Output")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.error(err)

        st.divider()
        st.subheader("Option 2 — Upload Jupyter Notebook")
        nb_file = st.file_uploader("Upload .ipynb", type=["ipynb"], key="nb")
        if nb_file:
            try:
                cells = notebook_code(nb_file.getvalue())
                st.success(f"Notebook loaded: {len(cells)} code cells found.")
                with st.expander("Preview notebook code"):
                    for i, cell in enumerate(cells, 1):
                        st.markdown(f"**Cell {i}**")
                        st.code(cell, language="python")

                if st.button("▶ Run Notebook", type="primary"):
                    ok, output, fig, err = run_code("\n\n".join(cells), df)
                    if ok:
                        st.success("Notebook executed successfully.")
                        if output: st.code(output)
                        if fig is not None:
                            st.subheader("Chart Output")
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.error(err)
            except Exception as e:
                st.error(f"Could not read notebook: {type(e).__name__}: {e}")

        st.info("For notebooks, upload the CSV used by the notebook first. It is available as df.")

elif lab == "SQL Lab":
    st.header("🗄️ SQL Practice Lab")
    csv_file = st.file_uploader("Upload CSV dataset", type=["csv"], key="sql")
    if csv_file:
        df = pd.read_csv(csv_file)
        st.dataframe(df.head(10), use_container_width=True)
        st.info("Your CSV is available as SQL table: sales")
        query = st.text_area("SQL Query", "SELECT * FROM sales LIMIT 10;", height=180)
        if st.button("▶ Run SQL", type="primary"):
            try:
                conn = sqlite3.connect(":memory:")
                df.to_sql("sales", conn, index=False, if_exists="replace")
                result = pd.read_sql_query(query, conn)
                conn.close()
                st.success(f"{len(result):,} rows returned.")
                st.dataframe(result, use_container_width=True)
                st.download_button("Download Result CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    "sql_result.csv", "text/csv")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
