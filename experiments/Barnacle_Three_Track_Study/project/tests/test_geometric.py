import unittest
import numpy as np
from barnacles.geometric import ring_strength,line_spans,longest_span,evidence,detect,GeometricConfig


class GeometricTests(unittest.TestCase):
    def test_separate_dark_pixels_do_not_add_up_to_a_long_slit(self):
        mask=np.array([[True,True,False,True,True,False,True,True]])
        self.assertAlmostEqual(float(longest_span(mask,.1)[0]),.1)

    def test_line_length_is_fraction_of_diameter_not_radius(self):
        g=np.ones((101,101),np.float32)
        g[49:52,36:65]=0 # 28 pixels between extreme centers = 70% of diameter40
        p=np.array([[50.,50.,20.,0.]])
        span=line_spans(g,p,np.array([20.]),contrasts=(.2,))
        self.assertGreaterEqual(float(span[0,0]),.675)
        self.assertLessEqual(float(span[0,0]),.725)
        g[:]=1;g[49:52,44:57]=0
        short=line_spans(g,p,np.array([20.]),contrasts=(.2,))
        self.assertLess(float(short[0,0]),.4)

    def test_ring_has_outward_contrast_and_uniform_patch_does_not(self):
        yy,xx=np.mgrid[:101,:101];r=np.hypot(xx-50,yy-50)
        g=(r>=20).astype(np.float32);p=np.array([[50.,50.,20.,0.]])
        self.assertGreater(np.mean(ring_strength(g,p,np.array([20.]))>=.2),.9)
        self.assertTrue(np.all(ring_strength(np.ones_like(g),p,np.array([20.]))==0))

    def test_wrong_tolerance_cache_is_rejected(self):
        ev=dict(radial_tolerance=np.array(.2))
        with self.assertRaises(ValueError):detect(np.empty((0,4)),ev,GeometricConfig())

    def test_radius_prior_applies_before_best_radius_selection(self):
        from dataclasses import replace
        p=np.array([[20.,20.,3.,1.]])
        ev=dict(radial_tolerance=np.array(.05),contrasts=np.array([.2]),multipliers=np.array([1.,2.]),
            ring_coverage=np.ones((1,2,1)),line_fraction=np.ones((1,2,1))*.8,ring_strength=np.ones((1,2)))
        cfg=replace(GeometricConfig(),min_radius=5.,max_radius=7.)
        det,_,_=detect(p,ev,cfg)
        self.assertEqual(float(det[0,2]),6.)

    def test_literal_annulus_rejects_edges_outside_five_percent(self):
        from ring_analysis.run_literal_edges import angular_edge_coverage
        angles=(np.arange(8)+.5)*2*np.pi/8
        points=10*np.column_stack((np.cos(angles),np.sin(angles)))
        centers=np.zeros((1,2));radii=np.array([[10.]])
        self.assertEqual(float(angular_edge_coverage(points,centers,radii,.05,bins=8)[0,0]),1.)
        self.assertEqual(float(angular_edge_coverage(points*1.06,centers,radii,.05,bins=8)[0,0]),0.)


if __name__=='__main__':unittest.main()
