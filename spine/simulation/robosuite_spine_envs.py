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
    """Scaling config for peg/nut size and clearance enforcement."""

    target_side_m: float = 0.02
    target_clearance_min_m: float = 0.0005
    target_clearance_max_m: float = 0.001
    target_clearance_m: float = 0.0008


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
        self._apply_target_side(xml_root)
        self._enforce_tight_tolerance(xml_root)
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

    def _apply_target_side(self, root: ET.Element) -> None:
        target_half = self.reality_scaling.target_side_m / 2.0
        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            if "square_peg" in name or name.startswith("peg"):
                size = _parse_vec(geom.get("size"))
                if size is None or len(size) != 3:
                    continue
                size[0] = max(
                    1e-4, target_half - self.reality_scaling.target_clearance_m / 2.0
                )
                size[1] = size[0]
                geom.set("size", _fmt_vec(size))
            if "square_nut" in name or "SquareNut" in name:
                size = _parse_vec(geom.get("size"))
                if size is None or len(size) != 3:
                    continue
                size[0] = target_half
                size[1] = target_half
                geom.set("size", _fmt_vec(size))

    def _enforce_tight_tolerance(self, root: ET.Element) -> None:
        """Tighten clearance between peg and nut to target gap (0.5mm-1.0mm)."""
        peg_geoms: list[ET.Element] = []
        nut_geoms: list[ET.Element] = []

        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            gtype = geom.get("type", "")
            if gtype != "box":
                continue
            if "square_peg" in name or name.startswith("peg"):
                peg_geoms.append(geom)
            if "square_nut" in name or "SquareNut" in name:
                nut_geoms.append(geom)

        if not peg_geoms or not nut_geoms:
            return

        # Use the first matching geom as reference.
        peg_size = _parse_vec(peg_geoms[0].get("size"))
        nut_size = _parse_vec(nut_geoms[0].get("size"))
        if peg_size is None or nut_size is None:
            return

        peg_half = float(peg_size[0])
        nut_half = float(nut_size[0])
        target_gap = float(self.reality_scaling.target_clearance_m)
        target_gap = max(
            self.reality_scaling.target_clearance_min_m,
            min(self.reality_scaling.target_clearance_max_m, target_gap),
        )

        target_half = (peg_half + nut_half) / 2.0
        peg_half_new = max(1e-4, target_half - target_gap / 2.0)
        nut_half_new = max(
            peg_half_new + target_gap / 2.0, target_half + target_gap / 2.0
        )

        for geom in peg_geoms:
            size = _parse_vec(geom.get("size"))
            if size is None or len(size) != 3:
                continue
            size[0] = peg_half_new
            size[1] = peg_half_new
            geom.set("size", _fmt_vec(size))

        for geom in nut_geoms:
            size = _parse_vec(geom.get("size"))
            if size is None or len(size) != 3:
                continue
            size[0] = nut_half_new
            size[1] = nut_half_new
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
        peg_body = _find_body(root, ("peg1", "square_peg"))
        riser_size = np.array([0.05, 0.05, 0.02])
        riser_offset = np.array([0.0, -0.14, -0.02])

        parent = peg_body if peg_body is not None else worldbody
        if peg_body is None:
            peg_pos = _find_body_pos(root, ("peg1", "square_peg"))
            if peg_pos is None:
                peg_pos = np.array([0.0, 0.0, self.table_offset[2]])
            riser_pos = peg_pos + riser_offset
        else:
            riser_pos = riser_offset

        riser = ET.SubElement(
            parent,
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
        tripod_body = _find_body(root, ("tripod_obj", "ring_tripod", "tripod", "ring"))
        bar_offset = np.array([-0.13, 0.0, -0.04])
        bar_size = (0.02, 0.10, 0.02)

        parent = tripod_body if tripod_body is not None else worldbody
        if tripod_body is None:
            hole_pos = _find_body_pos(
                root, ("tripod_obj", "ring_tripod", "tripod", "ring")
            )
            if hole_pos is None:
                hole_pos = np.array([0.0, 0.0, self.table_offset[2]])
            bar_pos = hole_pos + bar_offset
        else:
            bar_pos = bar_offset

        bar = ET.SubElement(
            parent,
            "geom",
            {
                "name": "fixture_rest_bar",
                "type": "box",
                "size": _fmt_vec(bar_size),
                "pos": _fmt_vec(bar_pos),
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
    body = _find_body(root, names)
    if body is None:
        return None
    pos = _parse_vec(body.get("pos"))
    if pos is None:
        return None
    return np.array(pos)


def _find_body(root: ET.Element, names: Iterable[str]) -> Optional[ET.Element]:
    for body in root.findall(".//body"):
        name = body.get("name", "")
        if any(token in name for token in names):
            return body
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
