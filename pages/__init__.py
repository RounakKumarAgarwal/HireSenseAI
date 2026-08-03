"""
pages package
=============
Each module in this package renders one screen of the HireSense AI
Streamlit app and exposes a single `render()` function, which
app.py calls based on the sidebar selection.

Note on filenames: the original spec used spaces in filenames
(e.g. "Resume Screening.py"), but Python module names cannot
contain spaces or be imported with `from pages import Resume Screening`.
We use underscores instead (resume_screening.py style, capitalized
to match the spec's intent) so the app can cleanly import
`from pages import Resume_Screening`.
"""
