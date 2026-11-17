"""Run the frozen confirmation pipeline; stop immediately on failure."""
import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for script,args in [('run_selfcal_confirmation.py',['score']),('analyze_selfcal_confirmation.py',[]),('validate_selfcal_confirmation.py',[])]:
 subprocess.run([sys.executable,str(R/'scripts'/script),*args],cwd=R,check=True)
