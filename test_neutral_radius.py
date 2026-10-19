"""A stronger blur is isolated from the original evidence and its cache."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image,ImageDraw,ImageChops
import run_web_eval


class NeutralRadius(unittest.TestCase):
    def test_control_changes_pixels_without_overwriting_default_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'page.png'
            im=Image.new('RGB',(96,96),'white');ImageDraw.Draw(im).rectangle((25,25,70,70),fill='black');im.save(source)
            blocks=[{'type':'I','image':str(source)}]
            with patch.object(run_web_eval,'ROOT',root):
                default=run_web_eval.neutralize(blocks,[-1],12)[0]['image']
                original=Path(default).read_bytes()
                control=run_web_eval.neutralize(blocks,[-1],24)[0]['image']
                self.assertEqual(Path(default).read_bytes(),original)
                self.assertNotEqual(default,control)
                with Image.open(default) as a,Image.open(control) as b:
                    self.assertEqual(a.size,b.size)
                    self.assertIsNotNone(ImageChops.difference(a,b).getbbox())
                self.assertEqual(run_web_eval.neutralize(blocks,[-1],24)[0]['image'],control)


if __name__=='__main__':unittest.main()
