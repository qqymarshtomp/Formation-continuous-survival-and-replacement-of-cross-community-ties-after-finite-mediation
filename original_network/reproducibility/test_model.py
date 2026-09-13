"""Scientific invariants and experimental-design tests (standard library)."""
import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
from model import Config,NetworkModel,run_withdrawal


class ModelTests(unittest.TestCase):
    def test_process_conservation_and_state_bounds(self):
        model=NetworkModel(Config(n=100,burnin=3,intervention=3,followup=4),5)
        for _ in range(100):
            row=model.step("context")
            self.assertEqual(row["count_residual"],0)
            self.assertLess(abs(row["weight_residual"]),1e-10)
            model.verify_state()

    def test_saved_state_reproduces_future(self):
        model=NetworkModel(Config(n=100),9)
        for _ in range(7):
            model.step("context")
        model.mark_withdrawal()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"checkpoint.npz"
            model.save(p)
            restored=NetworkModel.load(p)
        for _ in range(10):
            model.step("context"); restored.step("context")
        np.testing.assert_array_equal(model.w,restored.w)
        np.testing.assert_array_equal(model.x,restored.x)
        np.testing.assert_array_equal(model.r,restored.r)

    def test_decay_only_matches_exact_lifetime(self):
        model=NetworkModel(Config(n=100,decay=0.03),12)
        w0=model.w.copy()
        for t in range(1,100):
            model.step(natural_contact=False)
            expected=w0*(1-model.cfg.decay)**t
            expected[expected < model.cfg.threshold]=0
            np.testing.assert_allclose(model.w,expected,atol=1e-14)

    def test_zero_input_effect_controls_coincide(self):
        c=Config(n=100,frame_effect=0,action_effect=0)
        a,b=NetworkModel(c,7),NetworkModel(c,7)
        for _ in range(50):
            a.step("context");b.step("contact_only")
        np.testing.assert_array_equal(a.w,b.w)
        np.testing.assert_array_equal(a.r,b.r)

    def test_branches_share_exit_state(self):
        c=Config(n=100,burnin=4,intervention=5,followup=8)
        _,branches=run_withdrawal(c,3)
        for metric in ["b","strength","trust","opinion_gap"]:
            self.assertEqual(len(set(v[0][metric] for v in branches.values())),1)

    def test_no_natural_gain_cannot_create_or_strengthen(self):
        m=NetworkModel(Config(n=100),11)
        for _ in range(100):
            before=m.w.copy()
            result=m.step(natural_reinforcement=False,natural_birth=False)
            self.assertEqual(result["natural_births"],0)
            self.assertEqual(result["natural_weight"],0)
            self.assertTrue(np.all(m.w <= before))


if __name__ == "__main__":
    unittest.main(verbosity=2)
