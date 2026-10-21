"""Check tiling/label conversion independently of trained detector quality."""
import unittest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from barnacles.modern import windows,starts,choose_confidence,aperture_boxes
from barnacles.core import load_reference
from modern_analysis.train_detector import build_dataset
from modern_analysis.train_custom import validate_groups


class ModernTests(unittest.TestCase):
    def test_capture_groups_cannot_cross_splits(self):
        valid=[{'capture_group':str(i),'split':s} for i,s in enumerate(('train','val','test'))]
        validate_groups(valid)
        with self.assertRaises(ValueError):
            validate_groups(valid+[{'capture_group':'0','split':'test'}])

    def test_data_volume_subsets_are_nested_with_equal_training_slots(self):
        import json
        cases={k:dict(im=np.zeros((1024,1024,3),np.uint8),
            boxes=np.array([[10,10,20,20],[200,200,210,210]],np.float32)) for k in (1,2)}
        tr=[(k,(0,0,384,1024)) for k in (1,2)];va=[(k,(512,0,704,1024)) for k in (1,2)]
        sets=[]
        with tempfile.TemporaryDirectory() as td:
            for fraction in (.25,.5,1.):
                folder=Path(td)/str(fraction)
                build_dataset(cases,folder,tr,va,fraction)
                rows=json.loads((folder/'manifest.json').read_text())
                training=[r for r in rows if r['split']=='train']
                self.assertEqual(len(training),20)
                sets.append({r['source_observation'] for r in training})
        self.assertEqual([len(x) for x in sets],[4,10,20])
        self.assertTrue(sets[0]<sets[1]<sets[2])

    def test_box_decoding_preserves_legacy_centers(self):
        import cv2
        rgba=np.zeros((64,64,4),np.uint8)
        cv2.rectangle(rgba,(20,20),(30,30),(0,0,255,255),1)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'mask.png';Image.fromarray(rgba).save(path)
            b,c=aperture_boxes(path,np.eye(3),size=64)
            gt=load_reference(path,np.eye(3),size=64)
            self.assertEqual(len(b),1)
            np.testing.assert_allclose(c,gt[:,:2])
            np.testing.assert_allclose(c,[[25,25]])
            self.assertTrue(np.all(b[0,:2]<c[0]) and np.all(b[0,2:]>c[0]))

    def test_cores_partition_every_pixel_once(self):
        for bounds in [(0,0,1024,1024),(768,0,1024,1024),(512,0,704,1024)]:
            own=np.zeros((1024,1024),np.uint8)
            for w in windows(bounds):
                x0,y0,x1,y1=w['core']
                # Ownership uses coordinates at pixel centers; testing integers
                # matches the legacy center convention in rectified coordinates.
                yy,xx=np.mgrid[:1024,:1024]
                own[(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)]+=1
            x0,y0,x1,y1=bounds
            np.testing.assert_equal(own[y0:y1,x0:x1],1)
            self.assertEqual(int(own.sum()),(x1-x0)*(y1-y0))

    def test_context_cores_stay_inside_their_crop(self):
        for w in windows():
            crop,core=w['crop'],w['core']
            self.assertGreaterEqual(core[0],crop[0]);self.assertGreaterEqual(core[1],crop[1])
            self.assertLessEqual(core[2],crop[2]);self.assertLessEqual(core[3],crop[3])
        self.assertEqual(starts(0,640),[0,192,384])

    def test_threshold_selection_includes_misses(self):
        p=np.array([[10,10,5,.9],[50,50,5,.1]],np.float32)
        g=np.array([[10,10,5],[70,70,5]],np.float32)
        chosen,_=choose_confidence(p,p[:,3],g)
        self.assertEqual(chosen['tp'],1);self.assertEqual(chosen['fn'],1)
        self.assertAlmostEqual(chosen['f1'],2/3)


if __name__=='__main__':unittest.main()
