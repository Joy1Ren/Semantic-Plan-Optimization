"""What each query id asks. The wording here is the single source for the definitions: the
ground-truth labeling runner sends it to the frontier models (adding, for SBTi, what does not
count), and benchmark.yaml's task prompts carry the same text (kept in sync by hand).

  1  documents stating the company is a member of RE100
  2  documents stating the SBTi validated one of the company's emissions targets
  3  documents stating the company currently uses an internal carbon price
  4  companies with 1 to 4 work-related fatalities in the reporting period
  5  manufacturers with SBTi-validated targets                   (sector labels and 2)
  6  manufacturers that currently use an internal carbon price   (sector labels and 3)
  7  SBTi-validated targets AND NO internal carbon price         (2 and 3)
  8  an internal carbon price AND NO SBTi-validated target       (3 and 2)

Queries 1-3 are answered with a verbatim quote, graded with the answer; 4-8 are graded by F1 over
the returned documents. The sector labels (frontier full-document labels, NACE Rev. 2 sections)
are not a query of their own.
"""
from __future__ import annotations

from dataclasses import dataclass

# The source dataset's sector names (DataNeed/company-reports, version 2024_01_22), which are NACE
# Rev. 2 sections, with Rev. 2's official section titles shown in the prompt. administrative_support
# and arts_entertainment appear only in the dataset's earlier 2024_01_19 version; they are listed
# so every Rev. 2 section has a name. "other" is taken as section S, Other service activities.
# All 21 sections (A-U) are listed, including those no document here falls in, so the list is the
# full classification rather than this dataset's subset.
SECTORS = {
    "agriculture": "agriculture, forestry and fishing",
    "mining": "mining and quarrying",
    "manufacturing": "manufacturing",
    "energy_supply": "electricity, gas, steam and air conditioning supply",
    "water_supply": "water supply; sewerage, waste management and remediation activities",
    "construction": "construction",
    "trade_repair": "wholesale and retail trade; repair of motor vehicles and motorcycles",
    "transportation": "transportation and storage",
    "accommodation_food": "accommodation and food service activities",
    "information_communication": "information and communication",
    "finance_insurance": "financial and insurance activities",
    "real_estate": "real estate activities",
    "professional_scientific": "professional, scientific and technical activities",
    "administrative_support": "administrative and support service activities",
    "public_administration": "public administration and defence; compulsory social security",
    "education": "education",
    "health_social": "human health and social work activities",
    "arts_entertainment": "arts, entertainment and recreation",
    "other": "other service activities",
    "households": ("activities of households as employers; undifferentiated goods- and "
                   "services-producing activities of households for own use"),
    "extraterritorial": "activities of extraterritorial organisations and bodies",
}
# The source labels already use these names; only its one-off "forestry" (JD Irving) is renamed,
# to the section that contains it.
SOURCE_TO_SECTOR = {**{name: (name,) for name in SECTORS}, "forestry": ("agriculture",)}
SECTOR_LIST = "; ".join(f"{name} ({title})" for name, title in SECTORS.items())

PRIMARY_ACTIVITY = f"the company's primary economic activity, exactly one of: {SECTOR_LIST}"

# The condition each quote-graded question tests, keyed by question name (query ids are assigned
# in QUERIES and may be reordered without touching labels).
DEFINITIONS = {
    "sbti_validated": (
        "it states that the Science Based Targets initiative (SBTi) has validated (also called "
        "approved) an emissions reduction target of the company that issued the document."),
    "re100_member": (
        "it states that the company that issued the document is a member of RE100, the global "
        "initiative of companies committed to 100% renewable electricity."),
    # S&P Dow Jones Indices' definition without "future": there it describes the costs the price
    # stands for, not whether the price is in use; the question asks for current use separately.
    "internal_carbon_price": (
        "An internal carbon price is a monetary value assigned to greenhouse gas emissions as a "
        "means of accounting for the costs of climate change regulation or a company's climate "
        "action ambitions. The document qualifies if it states that the company that issued it "
        "currently uses an internal carbon price."),
    # asked of every document; the query then lists the companies above a threshold chosen from
    # the distribution of these counts
    "work_fatality": (
        "the number of work-related fatalities among the employees and contractors of the company "
        "that issued the document during the period the document reports on"),
}


@dataclass(frozen=True)
class Query:
    id: int
    name: str
    task: str        # one line, for reports
    metric: str      # benchmark.yaml's query_metrics name


# Query 4's range. A count is right only for employees AND contractors, in the reporting year.
FATALITY_RANGE = (1, 4)

QUERIES = [
    Query(1, "sustainability_re100_member",
          "documents stating the company is an RE100 member", "set_f1_quote_jaccard"),
    Query(2, "sustainability_sbti_validated",
          "documents stating the SBTi validated the company's target", "set_f1_quote_jaccard"),
    Query(3, "sustainability_internal_carbon_price",
          "documents stating the company currently uses an internal carbon price", "set_f1_quote_jaccard"),
    Query(4, "sustainability_fatalities_in_range",
          f"companies with {FATALITY_RANGE[0]} to {FATALITY_RANGE[1]} work-related fatalities", "set_f1"),
    Query(5, "sustainability_manufacturing_sbti",
          "manufacturers with SBTi-validated targets", "set_f1"),
    Query(6, "sustainability_manufacturing_carbon_price",
          "manufacturers that currently use an internal carbon price", "set_f1"),
    Query(7, "sustainability_sbti_no_carbon_price",
          "SBTi-validated targets and no internal carbon price", "set_f1"),
    Query(8, "sustainability_carbon_price_no_sbti",
          "an internal carbon price and no SBTi-validated target", "set_f1"),
]
QUERY_BY_ID = {query.id: query for query in QUERIES}
