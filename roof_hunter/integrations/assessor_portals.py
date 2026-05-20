"""County property-record portals for storm-lead enrichment (Texas CAD, Oklahoma assessor).

Texas: Central Appraisal District (CAD) — appraisal records and public search.
Oklahoma: County assessor offices (structures vary).

URLs verified against official county/CAD sites as of project integration; portals may move—
use ``home_url`` landing pages when ``property_search_url`` breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class CountyPropertyPortal:
    """Public-facing org that holds parcel / appraisal searchable records."""

    organization: str
    home_url: str
    property_search_url: str
    jurisdiction: str  # "texas_cad" | "ok_assessor"
    notes: str = ""

    def primary_search_hint(self) -> str:
        if self.jurisdiction == "texas_cad":
            return "Texas CAD: search by property address or owner name as published on portal."
        return "Oklahoma: use county assessor search or GIS linked from county site."


def _tx_cad(org: str, home: str, search: str, notes: str = "") -> CountyPropertyPortal:
    return CountyPropertyPortal(
        organization=org,
        home_url=home,
        property_search_url=search,
        jurisdiction="texas_cad",
        notes=notes,
    )


def _ok_assessor(org: str, home: str, search: str, notes: str = "") -> CountyPropertyPortal:
    return CountyPropertyPortal(
        organization=org,
        home_url=home,
        property_search_url=search,
        jurisdiction="ok_assessor",
        notes=notes,
    )


# Key: ("TX", "TARRANT") — county uppercased, no word "County"
_PORTALS: Dict[Tuple[str, str], CountyPropertyPortal] = {
    # Texas — counties appearing in roof_hunter SPC-derived lead extracts
    ("TX", "TARRANT"): _tx_cad(
        "Tarrant Appraisal District",
        "https://www.tad.org/",
        "https://tarrant.prodigycad.com/property-search",
    ),
    ("TX", "DALLAS"): _tx_cad(
        "Dallas Central Appraisal District",
        "https://www.dallascad.org/",
        "https://esearch.dallascad.org/SearchAddr.aspx",
    ),
    ("TX", "JOHNSON"): _tx_cad(
        "Johnson Central Appraisal District",
        "https://www.johnsoncad.com/",
        "https://esearch.johnsoncad.com/",
    ),
    ("TX", "WISE"): _tx_cad(
        "Wise Central Appraisal District",
        "https://www.wise-cad.com/",
        "https://esearch.wise-cad.com/",
    ),
    ("TX", "PARKER"): _tx_cad(
        "Parker Central Appraisal District",
        "https://www.parkercad.org/",
        "http://iswdataclient.azurewebsites.net/webSearchID.aspx?dbkey=parkercad&sdata=&stype=id",
        notes="ISW search hub; GIS also linked from parkercad.org.",
    ),
    ("TX", "JACK"): _tx_cad(
        "Jack County Appraisal District",
        "https://www.jackcad.org/",
        "https://jackcad.org/Home/Search",
    ),
    ("TX", "ROCKWALL"): _tx_cad(
        "Rockwall Central Appraisal District",
        "https://www.rockwallcad.com/",
        "https://rockwall.myprop.tax/search",
    ),
    ("TX", "GRAYSON"): _tx_cad(
        "Grayson Central Appraisal District",
        "https://www.graysonappraisal.org/",
        "https://esearch.graysonappraisal.org/",
        notes="Branded Grayson Appraisal; GIS at gis.bisclient.com/graysoncad/",
    ),
    ("TX", "HILL"): _tx_cad(
        "Hill Central Appraisal District",
        "https://www.hillcad.org/",
        "https://esearch.hillcad.org/",
    ),
    ("TX", "PALO PINTO"): _tx_cad(
        "Palo Pinto Central Appraisal District",
        "https://www.ppctad.org/",
        "https://propaccess.ppctad.org/client/",
    ),
    ("TX", "ELLIS"): _tx_cad(
        "Ellis Central Appraisal District",
        "https://www.elliscad.org/",
        "https://www.elliscad.org/",
        notes="Open Public Portal / property search from elliscad.org landing.",
    ),
    ("TX", "COLLIN"): _tx_cad(
        "Collin Central Appraisal District",
        "https://collincad.org/",
        "https://esearch.collincad.org/",
    ),
    ("TX", "FANNIN"): _tx_cad(
        "Fannin Central Appraisal District",
        "https://www.fannincad.org/",
        "https://www.fannincad.org/",
        notes="Use Property Search navigation on fannincad.org.",
    ),
    ("TX", "MONTAGUE"): _tx_cad(
        "Montague County Appraisal District",
        "https://www.montaguecad.net/",
        "http://iswdataclient.azurewebsites.net/webSearchID.aspx?dbkey=MONTAGUECAD&sdata=&stype=id",
    ),
    ("TX", "MCLENNAN"): _tx_cad(
        "McLennan Central Appraisal District",
        "https://www.mcad-tx.org/",
        "http://propaccess.mcad-tx.org/client/",
        notes="Waco-area CAD; branded MCAD.",
    ),
    ("TX", "CLAY"): _tx_cad(
        "Clay County Central Appraisal District",
        "https://www.claycad.org/",
        "https://www.public.claycad.org/",
        notes="If public subdomain fails, use property search linked from claycad.org.",
    ),
    ("TX", "HUNT"): _tx_cad(
        "Hunt County Central Appraisal District",
        "https://hunt-cad.org/",
        "https://public.hunt-cad.org/",
        notes="If public.* fails, use search from hunt-cad.org.",
    ),
    ("TX", "BOSQUE"): _tx_cad(
        "Bosque Central Appraisal District",
        "https://www.bosquecad.com/",
        "https://esearch.bosquecad.com/",
    ),
    ("TX", "HENDERSON"): _tx_cad(
        "Henderson Central Appraisal District",
        "https://henderson-cad.org/",
        "https://henderson-cad.org/",
        notes="Launch property search / esearch links from Henderson CAD homepage.",
    ),
    ("TX", "YOUNG"): _tx_cad(
        "Young Central Appraisal District",
        "https://youngcad.org/",
        "https://esearch.youngcad.org/",
    ),
    ("TX", "HOOD"): _tx_cad(
        "Hood Central Appraisal District",
        "https://hoodcad.net/",
        "https://hoodcad.publicaccessnow.com/Home.aspx",
        notes="If PublicAccess URL changes, start from hoodcad.net.",
    ),
    # Oklahoma — counties from SPC-derived extracts
    ("OK", "PONTOTOC"): _ok_assessor(
        "Pontotoc County Assessor",
        "https://pontotoc.okcounties.org/offices/assessor",
        "https://pontotoc.okcounties.org/offices/assessor",
        notes="OK Counties hub; follow links to hosted search if offered.",
    ),
    ("OK", "GARVIN"): _ok_assessor(
        "Garvin County Assessor",
        "https://www.gcaook.com/",
        "https://www.gcaook.com/search/",
    ),
    ("OK", "MURRAY"): _ok_assessor(
        "Murray County Assessor",
        "https://murray.okcounties.org/offices/assessor",
        "http://murray.oklahoma.usassessor.com/Shared/base/LiteSearch/Search.php",
        notes="Hosted assessor search; fallback to county office if URL changes.",
    ),
    ("OK", "SEMINOLE"): _ok_assessor(
        "Seminole County Assessor",
        "https://seminolecountyok.com/county-assessors-office",
        "https://seminolecountyok.com/county-assessors-office",
    ),
    ("OK", "CLEVELAND"): _ok_assessor(
        "Cleveland County Assessor",
        "https://www.clevelandcountyok.gov/Directory/Home/DepartmentListing?DID=28",
        "https://property.spatialest.com/ok/cleveland",
        notes="Hosted map/search; obey SpatialEst/site terms.",
    ),
    ("OK", "STEPHENS"): _ok_assessor(
        "Stephens County Assessor",
        "https://scaook.com/assessor/",
        "https://scaook.com/assessor/",
    ),
    ("OK", "COMANCHE"): _ok_assessor(
        "Comanche County Assessor",
        "https://comanchecoassessor.com/",
        "https://comanchecoassessor.com/",
        notes="Also https://www.comanchecountyok.gov/directory.aspx?did=7",
    ),
    ("OK", "POTTAWATOMIE"): _ok_assessor(
        "Pottawatomie County Assessor",
        "https://www.pottawatomiecountyok.gov/",
        "https://www.pottawatomiecountyok.gov/%5C176",
    ),
    ("OK", "GARFIELD"): _ok_assessor(
        "Garfield County Assessor",
        "https://garfield.okcounties.org/offices/assessor",
        "https://garfield.okcounties.org/offices/assessor",
    ),
    ("OK", "NOBLE"): _ok_assessor(
        "Noble County Assessor",
        "https://noble.okcounties.org/offices/assessor",
        "https://noble.okcounties.org/offices/assessor",
    ),
    ("OK", "PAYNE"): _ok_assessor(
        "Payne County Assessor",
        "https://payne.okcounties.org/offices/assessor",
        "https://payne.okcounties.org/offices/assessor",
    ),
    ("OK", "GRADY"): _ok_assessor(
        "Grady County Assessor",
        "https://grady.okcounties.org/offices/assessor",
        "https://grady.okcounties.org/offices/assessor",
    ),
    ("OK", "HUGHES"): _ok_assessor(
        "Hughes County Assessor",
        "https://hughes.okcounties.org/offices/assessor",
        "https://hughes.okcounties.org/offices/assessor",
    ),
}


def normalize_county_key(county: str) -> str:
    c = county.strip().upper()
    if c.endswith(" COUNTY"):
        c = c[: -len(" COUNTY")]
    return c.strip()


def get_county_property_portal(county: str, state: str) -> Optional[CountyPropertyPortal]:
    st = state.strip().upper()
    ck = normalize_county_key(county)
    return _PORTALS.get((st, ck))


def list_portals() -> Mapping[Tuple[str, str], CountyPropertyPortal]:
    return _PORTALS
