import time
import functools
import threading
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
import pandas as pd
# timeit decorator removed; ProgressLogger now handles elapsed timing and percentage updates

def get_sa_score(data: str) -> float:
    data = data.split(">>")[0]
    mol = Chem.MolFromSmiles(data)
    return sascorer.calculateScore(mol)

class ProgressLogger:
    """Thread-safe progress logger printing elapsed time and percentage updates.

    Prints whenever the percentage (rounded to `precision` decimals) changes, and always prints final state on completion.
    """
    def __init__(self, total, prefix="", precision: int = 1):
        self.total = int(total)
        self.processed = 0
        self.lock = threading.Lock()
        self.prefix = prefix
        self.start = time.perf_counter()
        self.precision = int(precision)
        self.last_shown = None

    def _format_elapsed(self, seconds: float) -> str:
        ms = int((seconds - int(seconds)) * 1000)
        s = int(seconds) % 60
        m = (int(seconds) // 60) % 60
        h = int(seconds) // 3600
        if h:
            return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
        else:
            return f"{m:d}:{s:02d}.{ms:03d}"

    def _percent(self) -> float:
        if self.total == 0:
            return 100.0
        return round((self.processed * 100.0) / self.total, self.precision)

    def update(self, n: int = 1):
        with self.lock:
            self.processed += n
            percent = self._percent()
            # print when the rounded percentage changes or when complete
            if self.last_shown is None or percent != self.last_shown or self.processed == self.total:
                self.last_shown = percent
                elapsed = time.perf_counter() - self.start
                elapsed_s = self._format_elapsed(elapsed)
                print(f"{self.prefix}{percent:.{self.precision}f}% ({self.processed}/{self.total}) elapsed: {elapsed_s}")

def run_sequential(smiles: list, precision: int = 1):
    """Run SA scoring sequentially over the provided SMILES list."""
    results = []
    logger = ProgressLogger(len(smiles), prefix='[seq] ', precision=precision)
    for smi in smiles:
        score = get_sa_score(smi)
        results.append((smi, score))
        logger.update()
    return results

def run_parallel(smiles: list, max_workers: int = None, precision: int = 1):
    """Run SA scoring in parallel using threads via asyncio + run_in_executor.

    If max_workers is provided, it will be used to create a ThreadPoolExecutor.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    logger = ProgressLogger(len(smiles), prefix='[par] ', precision=precision)

    def _worker(smi):
        score = get_sa_score(smi)
        logger.update()
        return score

    async def _run():
        loop = asyncio.get_running_loop()
        if max_workers is None:
            tasks = [loop.run_in_executor(None, _worker, smi) for smi in smiles]
            scores = await asyncio.gather(*tasks)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [loop.run_in_executor(executor, _worker, smi) for smi in smiles]
                scores = await asyncio.gather(*tasks)
        return list(zip(smiles, scores))

    return asyncio.run(_run())

def process_output(results, ):
    df = pd.DataFrame(results, columns=["SMILES_struct", "SA_Score"])
    return df

if __name__ == "__main__":
    # test_smiles = [
    #     "CCO",
    #     "CC(=O)O",
    #     "c1ccccc1",
    #     "CCN",
    #     "CCOCC",
    #     "C1CCCCC1",
    #     "CCOC(=O)C",
    #     "NCC(=O)O",
    #     "CCCl",
    #     "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    # ]

    # print("Running sequential...")
    # seq_res = run_sequential(test_smiles)
    # for smi, score in seq_res:
    #     print(f"{smi} -> SA: {score}")

    # print("\nRunning parallel (default threads)...")
    # par_res = run_parallel(test_smiles)
    # for smi, score in par_res:
    #     print(f"{smi} -> SA: {score}")

    from test_data import test_data

    smiles_data = [test[0] for test in test_data]

    del test_data

    print("\nRunning parallel (max_workers=5)...")
    par_res2 = run_parallel(smiles_data, max_workers=5)
    df_output = process_output(par_res2)
    df_output.to_csv("sa_scores_output.csv", index=False) 
    # for smi, score in par_res2:
    #     print(f"{smi} -> SA: {score}")
