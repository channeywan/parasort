#!/usr/bin/env python3
"""Check the requested E4b experiment coverage without running sorting code."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bench
import run_experiments as runner


class E4bMatrixTests(unittest.TestCase):
    def collect(self,quick):
        commands=[]
        def record(exe,n,dist='uniform',threads=1,procs=1,variant=None,extra=(),env=None,prefix=()):
            commands.append({'exe':exe,'n':n,'dist':dist,'threads':threads,'procs':procs,'variant':variant,'options':list(extra),'env':env})
        previous=Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(bench,'ROOT',Path(directory)), patch.object(bench,'run',side_effect=record), contextlib.redirect_stdout(io.StringIO()):
                bench.main(['E4b','--dry-run']+(['--quick'] if quick else []))
                self.assertEqual(list(Path(directory).iterdir()),[])
        finally:
            os.chdir(previous)
        return commands

    def test_both_scans_cover_required_parameters_in_both_modes(self):
        for quick,n,repeats in ((True,1<<18,2),(False,1<<28,5)):
            with self.subTest(quick=quick):
                commands=self.collect(quick)
                configs=[]
                for command in commands:
                    self.assertEqual(command['exe'],'psrs_omp')
                    self.assertEqual(command['variant'],'psrs')
                    self.assertEqual(command['n'],n)
                    self.assertEqual(command['procs'],1)
                    options=command['options']
                    k=int(options[options.index('--oversample')+1])
                    configs.append((command['dist'],command['threads'],k,'--no-tiebreak' not in options))
                self.assertEqual(len(configs),37)
                self.assertEqual(len(set(configs)),37,'the overlapping scan point must reuse its measurements')
                expected_oversample={(dist,64,k,True) for dist in ('uniform','normal','hotkeys') for k in (1,2,4,8,16,32)}
                expected_tiebreak={(dist,p,1,tie) for dist in ('hotkeys','few_unique') for p in (8,16,32,64,128) for tie in (True,False)}
                self.assertEqual(set(configs),expected_oversample|expected_tiebreak)
                self.assertEqual(bench.R,repeats)
                for command in commands:
                    self.assertEqual(command['env']['OMP_PLACES'],'threads' if command['threads']==128 else 'cores')

    def test_canonical_order_and_only_selection_include_e4b(self):
        import argparse
        selected=runner.selection('E13,E5,E4b,E4',argparse.ArgumentParser())
        self.assertEqual(selected,['E4','E4b','E5','E13'])
        self.assertEqual(runner.EXPERIMENTS.count('E4b'),1)
        self.assertEqual(runner.EXPERIMENTS[-1],'E13')


if __name__=='__main__':unittest.main()
