"""Check the shared PyCharm run configurations against the PythonCode entry points."""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / ".run"


def options(path):
    configuration = ET.parse(path).getroot().find("configuration")
    values = {option.get("name"): option.get("value") for option in configuration.findall("option")}
    values["env"] = {env.get("name"): env.get("value") for env in configuration.find("envs")}
    values["name"] = configuration.get("name")
    return values


class RunConfigurationTests(unittest.TestCase):
    def test_configurations_point_to_existing_entry_points(self):
        files = sorted(RUN_DIR.glob("*.run.xml"))
        self.assertGreaterEqual(len(files), 7)
        names = set()
        for path in files:
            with self.subTest(path=path.name):
                values = options(path)
                names.add(values["name"])
                self.assertEqual(values["WORKING_DIRECTORY"], "$PROJECT_DIR$/PythonCode")
                self.assertEqual(values["env"].get("PYTHONUTF8"), "1")
                if values["MODULE_MODE"] == "true":
                    self.assertEqual(values["SCRIPT_NAME"], "unittest")
                    self.assertTrue((ROOT / "PythonCode" / "tests").is_dir())
                    continue
                script = values["SCRIPT_NAME"].replace("$PROJECT_DIR$", str(ROOT))
                self.assertTrue(Path(script).is_file(), script)
                if script.endswith("02_SOP_Experiments.py"):
                    parameters = values["PARAMETERS"].split()
                    self.assertIn(parameters[parameters.index("--method") + 1], {"baselines", "supcon"})
        self.assertEqual(len(names), len(files), "Run configuration names must be unique.")


if __name__ == "__main__":
    unittest.main()
