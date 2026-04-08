"""Add position or motor actuators to a MuJoCo XML that only has joints.

Reads all <joint> elements from the XML, generates corresponding <actuator>
entries (position or motor), and writes the modified XML to a new file.

Actuator naming convention:  J_XXX  ->  M_XXX  (matching project style).

Usage:
    python add_actuators.py
"""

import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Configuration — edit these directly, no CLI arguments needed
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

# Input / output XML paths
INPUT_XML = _THIS_DIR.parent / "models" / "xmls" / "ftl_xleo.xml"
OUTPUT_XML = None  # None = auto: <input>_<type>.xml

# Actuator type: "motor" (torque) or "position" (PD servo)
ACTUATOR_TYPE = "motor"

# Position actuator gains (only used when ACTUATOR_TYPE == "position")
KP = 100.0
KV = 10.0

# Whether to add forcerange from joint actuatorfrcrange
ADD_FORCERANGE = True

# Regex filter for joint names (None = all joints)
#   e.g. "F[012]_[LR]" for finger joints only
JOINT_FILTER = None

# Dry-run: only print the <actuator> block, don't write file
DRY_RUN = False


# ---------------------------------------------------------------------------
# Joint collector
# ---------------------------------------------------------------------------


def _collect_joints(root: ET.Element) -> list[dict]:
    """Walk the XML tree and collect all <joint> elements with their attrs."""
    joints = []
    for joint_elem in root.iter("joint"):
        attrs = dict(joint_elem.attrib)
        name = attrs.get("name")
        if name is None:
            continue
        joints.append(attrs)
    return joints


# ---------------------------------------------------------------------------
# Actuator element builders
# ---------------------------------------------------------------------------


def _actuator_name(joint_name: str) -> str:
    """J_XXX -> M_XXX (project convention)."""
    if joint_name.startswith("J_"):
        return "M_" + joint_name[2:]
    return "M_" + joint_name


def _ctrlrange_from_joint(joint_attrs: dict) -> str | None:
    """Derive ctrlrange from joint range attribute."""
    return joint_attrs.get("range")


def _forcerange_from_joint(joint_attrs: dict) -> str | None:
    """Derive forcerange from joint actuatorfrcrange attribute."""
    return joint_attrs.get("actuatorfrcrange")


def build_motor_element(joint_attrs: dict) -> ET.Element:
    """Build a <motor> actuator element for the given joint."""
    jname = joint_attrs["name"]
    elem = ET.Element("motor")
    elem.set("name", _actuator_name(jname))
    elem.set("joint", jname)

    cr = _ctrlrange_from_joint(joint_attrs)
    if cr is not None:
        elem.set("ctrlrange", cr)

    if ADD_FORCERANGE:
        fr = _forcerange_from_joint(joint_attrs)
        if fr is not None:
            elem.set("forcerange", fr)

    return elem


def build_position_element(joint_attrs: dict) -> ET.Element:
    """Build a <position> actuator element for the given joint."""
    jname = joint_attrs["name"]
    elem = ET.Element("position")
    elem.set("name", _actuator_name(jname))
    elem.set("joint", jname)

    cr = _ctrlrange_from_joint(joint_attrs)
    if cr is not None:
        elem.set("ctrlrange", cr)

    elem.set("kp", str(KP))
    elem.set("kv", str(KV))

    if ADD_FORCERANGE:
        fr = _forcerange_from_joint(joint_attrs)
        if fr is not None:
            elem.set("forcerange", fr)

    return elem


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _indent(elem: ET.Element, level: int = 0, indent_str: str = "  ") -> None:
    """In-place recursive indentation."""
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1, indent_str)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def actuator_block_to_string(actuator_elems: list[ET.Element]) -> str:
    """Render a list of actuator elements as a pretty XML <actuator> block."""
    wrapper = ET.Element("actuator")
    for ae in actuator_elems:
        wrapper.append(ae)
    _indent(wrapper, level=1)
    wrapper.tail = "\n"
    return ET.tostring(wrapper, encoding="unicode")


# ---------------------------------------------------------------------------
# Option block
# ---------------------------------------------------------------------------


def _ensure_option(root: ET.Element) -> None:
    """Ensure <option> element exists with fixed simulation settings.

    Produces:
        <option timestep="0.002" integrator="Euler">
          <flag eulerdamp="disable"/>
        </option>
    """
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)

    option.set("timestep", "0.002")
    option.set("integrator", "Euler")

    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("eulerdamp", "disable")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main():
    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    # Add/update <option> section
    _ensure_option(root)

    # Collect joints
    joints = _collect_joints(root)
    if not joints:
        print("No <joint> elements found in the XML.", file=sys.stderr)
        sys.exit(1)

    # Optional regex filter
    if JOINT_FILTER:
        pat = re.compile(JOINT_FILTER)
        joints = [j for j in joints if pat.search(j["name"])]
        if not joints:
            print(f"No joints matched filter '{JOINT_FILTER}'.", file=sys.stderr)
            sys.exit(1)

    # Build actuator elements
    actuator_elems: list[ET.Element] = []
    for jattrs in joints:
        if ACTUATOR_TYPE == "position":
            ae = build_position_element(jattrs)
        else:
            ae = build_motor_element(jattrs)
        actuator_elems.append(ae)

    block_str = actuator_block_to_string(actuator_elems)

    print(f"Generated {len(actuator_elems)} {ACTUATOR_TYPE} actuators:")
    for ae in actuator_elems:
        print(f"  {ae.get('name'):40s} -> joint={ae.get('joint')}")

    if DRY_RUN:
        print("\n--- Actuator block (dry-run) ---")
        print(block_str)
        return

    # Check if <actuator> already exists
    existing = root.find("actuator")
    if existing is not None:
        print(
            "\nWARNING: <actuator> section already exists. Replacing it.",
            file=sys.stderr,
        )
        root.remove(existing)

    # Insert <actuator> after </worldbody>
    actuator_section = ET.Element("actuator")
    for ae in actuator_elems:
        actuator_section.append(ae)

    children = list(root)
    insert_idx = len(children)
    for i, child in enumerate(children):
        if child.tag == "worldbody":
            insert_idx = i + 1
            break
    root.insert(insert_idx, actuator_section)

    # Determine output path
    output_path = OUTPUT_XML
    if output_path is None:
        stem = INPUT_XML.stem
        suffix = INPUT_XML.suffix
        tag = "position" if ACTUATOR_TYPE == "position" else "motor"
        output_path = INPUT_XML.parent / f"{stem}_{tag}{suffix}"

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=False)

    print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
