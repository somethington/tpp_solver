"""Entry point for the TPP Solver Streamlit app.

Run with:  streamlit run tpp_solver_mt.py

The implementation lives in the ``tpp_solver`` package; this module is a thin
launcher kept at the repository root so the existing run command and Docker
entrypoint continue to work.
"""
from tpp_solver.ui import main

if __name__ == "__main__":
    main()
