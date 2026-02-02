"""Robosuite env extensions for SPINE contact-rich experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import xml.etree.ElementTree as ET

import numpy as np

try:
    from mimicgen.envs.robosuite.nut_assembly import Square_D0
    from mimicgen.envs.robosuite.threading import Threading_D0
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "mimicgen (robosuite branch) is required for SPINE simulation envs."
    ) from exc


@dataclass
class ContactTuning:
    """Contact parameters for stiff, high-friction bracing."""

    condim: int = 3
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    solref: tuple[float, float] = (0.004, 1.0)
    solimp: tuple[float, float, float] = (0.9, 0.95, 0.001)


@dataclass
class RealityScaling:
    """Scaling config for peg/nut size and tolerance."""

    target_side_m: float = 0.02
    tolerance_m: float = 0.001


class SpineXMLMixin:
    """Mixin that edits robosuite XML to inject SPINE fixtures and physics."""

    contact_tuning: ContactTuning = ContactTuning()
    reality_scaling: RealityScaling = RealityScaling()

    def edit_model_xml(self, xml_str: str) -> str:
        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        self._modify_xml_for_reality(root)
        self._override_contacts(root)
        return ET.tostring(root, encoding="unicode")

    # ---- Public hook requested in spec ----
    def _modify_xml_for_reality(self, xml_root: ET.Element) -> None:
        """Applies scaling, tolerance, and friction overrides to task assets."""
        self._scale_nut_and_peg(xml_root)
        self._override_friction(xml_root)

    # ---- Helpers ----
    def _override_contacts(self, root: ET.Element) -> None:
        """Enable contact for wrist / forearm against table + fixtures."""
        self._remove_contact_excludes(root, {"robot0_link6", "robot0_link7"})
        self._set_geom_contact_params(
            root, name_filters=("table", "fixture"), tuning=self.contact_tuning
        )
        self._set_geom_contact_params(
            root,
            name_filters=("robot0_link6", "robot0_link7"),
            tuning=self.contact_tuning,
        )

    def _scale_nut_and_peg(self, root: ET.Element) -> None:
        target_half = self.reality_scaling.target_side_m / 2.0
        peg_half = target_half + self.reality_scaling.tolerance_m / 2.0
        nut_half = target_half + self.reality_scaling.tolerance_m

        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            gtype = geom.get("type", "")
            if gtype != "box":
                continue
            if "square_peg" in name or name.startswith("peg"):
                size = _parse_vec(geom.get("size"))
                if size is None or len(size) != 3:
                    continue
                size[0] = peg_half
                size[1] = peg_half
                geom.set("size", _fmt_vec(size))
            if "square_nut" in name or "SquareNut" in name:
                size = _parse_vec(geom.get("size"))
                if size is None or len(size) != 3:
                    continue
                size[0] = nut_half
                size[1] = nut_half
                geom.set("size", _fmt_vec(size))

    def _override_friction(self, root: ET.Element) -> None:
        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            if any(token in name for token in ("square_nut", "SquareNut", "peg")):
                geom.set("friction", "0.3 0.005 0.0001")
            if "table" in name:
                geom.set("friction", "1.0 0.005 0.0001")

    def _remove_contact_excludes(self, root: ET.Element, robot_links: set[str]) -> None:
        contact = root.find("contact")
        if contact is None:
            return
        for exclude in list(contact.findall("exclude")):
            body1 = exclude.get("body1", "")
            body2 = exclude.get("body2", "")
            if body1 in robot_links or body2 in robot_links:
                contact.remove(exclude)

    def _set_geom_contact_params(
        self, root: ET.Element, name_filters: Iterable[str], tuning: ContactTuning
    ) -> None:
        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            if not any(token in name for token in name_filters):
                continue
            geom.set("condim", str(tuning.condim))
            geom.set("friction", _fmt_vec(tuning.friction))
            geom.set("solref", _fmt_vec(tuning.solref))
            geom.set("solimp", _fmt_vec(tuning.solimp))


class SpineNutAssemblySquare(SpineXMLMixin, Square_D0):
    """NutAssemblySquare with SPINE riser and contact tuning."""

    def edit_model_xml(self, xml_str: str) -> str:
        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        self._modify_xml_for_reality(root)
        self._add_riser(root)
        self._override_contacts(root)
        return ET.tostring(root, encoding="unicode")

    def _add_riser(self, root: ET.Element) -> None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        table_z = self.table_offset[2]
        peg_pos = _find_body_pos(root, ("peg1", "square_peg"))
        if peg_pos is None:
            peg_pos = np.array([0.0, 0.0, table_z])
        offset = np.array([0.08, 0.0, 0.0])
        riser_size = np.array([0.04, 0.04, 0.02])
        riser_pos = peg_pos + offset
        riser_pos[2] = table_z + riser_size[2]
        riser = ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "fixture_riser",
                "type": "box",
                "size": _fmt_vec(riser_size),
                "pos": _fmt_vec(riser_pos),
                "rgba": "0.6 0.6 0.6 1",
            },
        )
        _apply_contact_params(riser, self.contact_tuning)


class SpineThreading(SpineXMLMixin, Threading_D0):
    """Threading task with SPINE rest bar and contact tuning."""

    def edit_model_xml(self, xml_str: str) -> str:
        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        self._modify_xml_for_reality(root)
        self._add_rest_bar(root)
        self._override_contacts(root)
        return ET.tostring(root, encoding="unicode")

    def _add_rest_bar(self, root: ET.Element) -> None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        table_z = self.table_offset[2]
        tripod_pos = _find_body_pos(root, ("ring_tripod", "tripod", "ring"))
        if tripod_pos is None:
            tripod_pos = np.array([0.0, 0.0, table_z])
        bar_pos = tripod_pos + np.array([0.0, -0.06, 0.08])
        bar = ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "fixture_rest_bar",
                "type": "cylinder",
                "size": _fmt_vec((0.01, 0.06)),
                "pos": _fmt_vec(bar_pos),
                "quat": "0.7071 0 0.7071 0",
                "rgba": "0.5 0.5 0.5 1",
            },
        )
        _apply_contact_params(bar, self.contact_tuning)


def _apply_contact_params(geom: ET.Element, tuning: ContactTuning) -> None:
    geom.set("condim", str(tuning.condim))
    geom.set("friction", _fmt_vec(tuning.friction))
    geom.set("solref", _fmt_vec(tuning.solref))
    geom.set("solimp", _fmt_vec(tuning.solimp))


def _find_body_pos(root: ET.Element, names: Iterable[str]) -> Optional[np.ndarray]:
    for body in root.findall(".//body"):
        name = body.get("name", "")
        if any(token in name for token in names):
            pos = _parse_vec(body.get("pos"))
            if pos is not None:
                return np.array(pos)
    return None


def _parse_vec(raw: Optional[str]) -> Optional[list[float]]:
    if raw is None:
        return None
    try:
        return [float(x) for x in raw.split()]
    except ValueError:
        return None


def _fmt_vec(values: Iterable[float]) -> str:
    return " ".join(f"{v:.5f}" for v in values)


__all__ = ["SpineNutAssemblySquare", "SpineThreading"]
