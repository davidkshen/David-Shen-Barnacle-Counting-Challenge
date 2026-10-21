"""Numerical geometry and evaluation checks, independent of measured scores."""
import unittest
import numpy as np
from barnacles.core import select_detections,normalized_gray,reference_targets
from barnacles.elliptical import (axes,coordinates,extract,EllipseConfig,augment,
    classifier_features,soft_rule_scores,split_membership)


class EllipticalTests(unittest.TestCase):
    def test_area_equivalent_shapes(self):
        a,b=axes(8,.5,2);self.assertAlmostEqual(np.pi*a*b,np.pi*64)
        a,b=axes(8,.5,1);self.assertAlmostEqual(2*a*b,np.pi*64)
        self.assertAlmostEqual(b/a,.5)

    def test_ellipse_orientation(self):
        p=np.array([[30,30,8,1]],np.float32)
        x,y=coordinates(p,.5,2,1,0,24)
        xx,yy=coordinates(p,.5,2,1,np.pi/2,24)
        np.testing.assert_allclose(xx,-y,atol=1e-5)
        np.testing.assert_allclose(yy,x,atol=1e-5)

    def test_radius_and_positions_scale_together(self):
        rgb=np.zeros((100,100,3),np.uint8);p=np.array([[20.,30.,8.,1.]],np.float32)
        im,pp=augment(rgb,p,'scale_0.85')
        self.assertEqual(im.shape[:2],(85,85))
        np.testing.assert_allclose(pp[0,:2],(p[0,:2]+.5)*.85-.5,atol=3e-6)
        self.assertAlmostEqual(float(pp[0,2]),6.8,places=5)
        np.testing.assert_equal(p[0,:3],[20,30,8])

    def test_blank_has_no_structural_evidence(self):
        cfg=EllipseConfig(ratios=(1.,),powers=(2.,),multipliers=(1.,),tolerances=(.1,))
        p=np.array([[32,32,8,1]],np.float32)
        b,s=extract(np.zeros((64,64),np.float32),p,cfg)
        # Sign agreement may be zero as well; every magnitude/support is zero.
        np.testing.assert_equal(b,0)
        score,_=soft_rule_scores(b,p,s,family='circles',tolerance=.1,size_weight=0)
        np.testing.assert_equal(score,0)

    def test_synthetic_elliptical_rim(self):
        yy,xx=np.mgrid[:81,:81];a,b=axes(10,.5)
        rho=np.sqrt(((xx-40)/a)**2+((yy-40)/b)**2)
        gray=np.where(rho>1,1.,-1.).astype(np.float32)
        p=np.array([[40,40,10,1]],np.float32)
        cfg=EllipseConfig(ratios=(.5,1.),powers=(2.,),multipliers=(1.,),tolerances=(.1,))
        bank,s=extract(gray,p,cfg)
        self.assertGreater(bank[0,0,0],bank[0,1,0])
        self.assertGreater(bank[0,0,2],.8)

    def test_nms_can_be_calibrated_once(self):
        p=np.array([[0,0,4,0],[1,0,4,0],[12,0,4,0],[13,0,4,0]],np.float32)
        score=np.array([.9,.6,.7,.2])
        full=select_detections(p,score,-1)
        for t in (.1,.5,.65,.8,.95):
            np.testing.assert_array_equal(select_detections(p,score,t),full[score[full]>=t])

    def test_split_support_and_feature_families(self):
        p=np.array([[367,40,17,1],[500,40,17,1],[656,40,17,1]],np.float32)
        tr,va=split_membership(p)
        np.testing.assert_equal(tr,[True,False,False]);np.testing.assert_equal(va,[False,False,True])
        # Maximum finite feature support in original pixels at the smallest zoom.
        major,_=axes(17*1.4,.5,1)
        support=(major*1.37+48+3)/.85
        self.assertLess(2*support,288)
        specs=[dict(power=1,ratio=.5),dict(power=2,ratio=.5),dict(power=2,ratio=1)]
        bank=np.zeros((3,3,12))
        self.assertEqual(classifier_features(bank,p,specs,'circles').shape,(3,13))
        self.assertEqual(classifier_features(bank,p,specs,'ellipses').shape,(3,25))
        self.assertEqual(classifier_features(bank,p,specs,'all').shape,(3,37))


if __name__=='__main__':unittest.main()
