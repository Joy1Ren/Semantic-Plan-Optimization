"""Build a clean CUAD ground-truth CSV: one row per contract, two columns per category.

Spans and the Yes/No answers come from CUADv1.json, the SQuAD-format CUAD release, whose spans
are verbatim substrings of the contract. CUADv1.json carries no normalized answers, so the
"-Answer" strings of the eight answer categories are read from CUAD's master_clauses.csv (the
official release file; docetl ships a copy) and parsed into JSON.

Columns of cuad_gt.csv:
  name                  CUADv1 title; equals the docetl / idx_to_name.json filename minus ".txt"
  unusable              {category: why its ground truth cannot be graded for this contract}, empty
                        for most rows. Skip a category for a contract listed here. The key
                        "number_parties" disqualifies only the count, not the Parties spans
  {category}            JSON list of the verbatim spans, in document order
  {category} Answer     "Yes" / "No" for the clause categories, otherwise JSON:
    Document Name       "MARKETING AFFILIATE AGREEMENT"
    Parties             {"number_parties": 2, "parties": ["Birch First Global Investments Inc.",
                        "Mount Knowledge Holdings Inc."]}; number_parties is null when one answer
                        entry covers several parties (see the issues file). Grade names against the
                        Parties spans, which hold each name and each defined term separately
    Agreement / Effective / Expiration Date
                        [{"month": "05", "day": "08", "year": "2014"}]; a "?" is a digit left
                        blank or redacted in the contract ("year": "19??"), so grade only the
                        parts that are known. Expiration Date may hold the item "perpetual", and
                        Effective Date the item "relative" (the contract ties it to an event such
                        as execution or a closing rather than to a calendar date)
    Renewal Term        [{"renewals": 1 | 2 | "unlimited", "length": 1, "unit": "years"}] or
                        ["perpetual"]; "successive" in CUAD means "unlimited". A few answers give
                        the renewal's end date instead, and hold a date item
    Notice Period To Terminate Renewal
                        [{"length": 30, "unit": "days"}]
    Governing Law       ["Ontario"] -- the most specific place named, so a model answering
                        "Ontario, Canada" or "Republic of Kazakhstan" still contains it; a list when
                        the contract names several. A law that is not a place is "non-place"
  An empty list means the contract has no answer for that category. A value the contract states
  but the filing blacks out is "redacted" -- a valid answer, not a gap: the clause was found and
  its value is withheld. Date digits blacked out or left blank are "?" ("year": "19??").

cuad_gt_issues.csv lists every (contract, category) whose ground truth cannot be checked cleanly
against the contract text, one row per issue. See ISSUE_CODES.
"""
from __future__ import annotations

import csv
import difflib
import json
import re

import pandas as pd

from agent_cost_model.experiments.cuad.paths import BENCHMARK_DIR, GROUND_TRUTH_DIR, cuad_ground_truth_csv

CUAD_JSON = BENCHMARK_DIR / "CUADv1.json"
CATEGORY_DESCRIPTIONS_CSV = BENCHMARK_DIR / "category_descriptions.csv"
OUTPUT_CSV = GROUND_TRUTH_DIR / "cuad_gt.csv"
ISSUES_CSV = GROUND_TRUTH_DIR / "cuad_gt_issues.csv"

DATE_CATEGORIES = ["Agreement Date", "Effective Date", "Expiration Date"]
ANSWER_CATEGORIES = ["Document Name", "Parties", *DATE_CATEGORIES, "Renewal Term",
                     "Notice Period to Terminate Renewal", "Governing Law"]

# The contract states the value but the filing blacks it out ("[***]", "[]"). Recorded as an
# answer of its own: finding the clause and reporting that its value is withheld is correct.
REDACTED = "redacted"
# A governing law that is not a place at all: "the state in which the breach occurs", a statute.
# Judge these on the spans only.
NON_PLACE = "non-place"
# The contract sets the date by an event (execution, a closing) or by a term running from one,
# rather than stating a calendar date. Answering "relative" is correct; no arithmetic is wanted.
RELATIVE = "relative"
# The contract has the clause but states no length or date for it ("the parties may negotiate an
# extension", "until Acceptance of all Deliverables"). Answering "unspecified" is correct.
UNSPECIFIED = "unspecified"

ISSUE_CODES = {
    "answer_without_span": "the answer has no supporting span in the contract",
    "span_without_answer": "spans exist but annotators gave no answer (e.g. a term relative to an event)",
    "yes_no_differs_from_csv": "master_clauses.csv's Yes/No disagrees with CUADv1.json's spans (json is used)",
    "relative_expiration": "the contract states no expiration date, so the answer is the term itself (or unspecified) rather than a date the annotators computed",
    "term_unclear": "the term the span writes and the term CUAD's computed date implies disagree",
    "computed_date": "the date's year is in no span: annotators took it from elsewhere in the contract or computed it (effective date + term)",
    "multiple_values": "the answer holds more than one value",
    "day_month_swapped": "the raw date was day-first and was reordered",
    "date_differs_from_span": "the spans write out a full date, but not the answer's",
    "typo_fixed": "a misspelled word in a party name was replaced by the contract's spelling",
    "party_not_in_spans": "a party name matches none of the Parties spans",
    "number_parties_unclear": "one answer entry covers several parties, so they cannot be counted",
    "not_a_jurisdiction": "a governing-law item is not a place",
    "override": "the raw answer was replaced by hand (see OVERRIDES)",
    "questionable": "the annotation looks wrong on reading the span (see FLAGS)",
}

# Issues that leave a category's ground truth unusable for that contract, and why. Everything
# else in the issues file is a note, not a disqualification.
UNUSABLE_ISSUES = {
    "answer_without_span": "CUAD gives an answer but highlights no supporting text",
    "span_without_answer": "CUAD highlights text but gives no answer",
    "computed_date": "the date is written nowhere in the spans; CUAD took it from elsewhere in the contract",
    "date_differs_from_span": "the spans write a different date than the answer",
    "term_unclear": "the term the span states and the term CUAD's date implies disagree",
    "number_parties_unclear": "one answer entry covers several parties, so they cannot be counted",
    "yes_no_differs_from_csv": "master_clauses.csv and the CUADv1 spans disagree on Yes/No",
    "questionable": "the annotation does not match what the span says",
    "different_values": "the answer holds two different values for one question",
}
# number_parties is graded separately from the Parties spans, so it is disqualified on its own.
UNUSABLE_KEYS = {"number_parties_unclear": "number_parties"}

# Contracts unusable for a reason no issue code catches.
UNUSABLE = {
    ("OPERALTD_04_30_2020-EX-4.14-SERVICE AGREEMENT", "Renewal Term"):
        "the Renewal Term spans are three chunks of the data-protection clause; nothing in them is about renewal",
    ("TubeMediaCorp_20060310_8-K_EX-10.1_513921_EX-10.1_Affiliate Agreement", "Renewal Term"):
        "CUAD's '4 years, 6 months' mixes the renewal term with the notice period",
    ("FulucaiProductionsLtd_20131223_10-Q_EX-10.9_8368347_EX-10.9_Content License Agreement",
     "Renewal Term"): "CUAD's 'perpetual, 11/15/2014' mixes the license term with a commencement date",
    ("MANUFACTURERSSERVICESLTD_06_05_2000-EX-10.14-OUTSOURCING AGREEMENT", "Renewal Term"):
        "the filing holds two agreements with different renewal terms (12 months, and 6 months for "
        "the Attachment), so one answer cannot cover it",
    # Renewal counts the span does not settle: how many times it renews cannot be read off the
    # text, so no answer to "unlimited or once" is gradable.
    ("AMERICANPHYSICIANSCAPITALINC_03_31_2003-EX-10.26-AGENCY AGREEMENT", "Renewal Term"):
        "\"subject to any automatic renewal or extension for one year as required by law\" does not "
        "say how many times it renews",
    ("HEMISPHERX - Sales, Marketing, Distribution, and Supply Agreement", "Renewal Term"):
        "\"an automatic 2 year term extensions\" reads as one extension and as repeating ones",
    ("RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE AGREEMENT", "Renewal Term"):
        "\"renewed automatically for another five (5) years\" reads as one renewal and as evergreen",
    ("SPHERE3DCORP_06_24_2020-EX-10.12-CONSULTING AGREEMENT", "Renewal Term"):
        "\"will automatically extend for an additional month of service\" reads as one month and as "
        "month after month",
    ("2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement", "Renewal Term"):
        "\"automatically renewed for another one (1) year\" reads as one renewal and as evergreen",
    ("TodosMedicalLtd_20190328_20-F_EX-4.10_11587157_EX-4.10_Marketing Agreement_ Reseller Agreement",
     "Renewal Term"):
        "the span renews 5 years once and then 2 years \"at the end of each renewal term\"; CUAD "
        "recorded both as single renewals",
    ("SoupmanInc_20150814_8-K_EX-10.1_9230148_EX-10.1_Franchise Agreement3", "Parties"):
        "no parties annotated: neither spans nor an answer",
}

# Answers their own span plainly contradicts: a date written out in the span, or a date the span
# states and the answer left blank. Only such clear errors are corrected -- an answer that is
# merely oddly formatted or debatable is kept as CUAD wrote it and reported in the issues file.
# Keyed by (CUADv1 title, category); the value replaces the master_clauses.csv answer.
OVERRIDES = {
    # The span is "May 25th, 2018"; the raw answer has 5/25/08.
    ("PAXMEDICA,INC_07_02_2020-EX-10.12-Master Service Agreement", "Agreement Date"): "5/25/18",
    # The span is "7th day of September, 1999"; the raw answer has 7/7/99.
    ("LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT", "Agreement Date"): "9/7/99",
    # "This Agreement shall commence on March 15, 2018"; the raw answer has 3/14/18.
    ("FreezeTagInc_20180411_8-K_EX-10.1_11139603_EX-10.1_Sponsorship Agreement", "Effective Date"): "3/15/18",
    # Dates their own span contradicts, or states while the answer left it blank.
    ("GluMobileInc_20070319_S-1A_EX-10.09_436630_EX-10.09_Content License Agreement2",
     "Agreement Date"): "11/11/05",                       # span: "November 11, 2005"
    ("BABCOCK_WILCOXENTERPRISES,INC_08_04_2015-EX-10.17-INTELLECTUAL PROPERTY AGREEMENT between "
     "THE BABCOCK _ WILCOX COMPANY and BABCOCK _ WILCOX ENTERPRISES, INC.",
     "Effective Date"): "6/26/15",                        # span: "June 26, 2015"
    ("PAXMEDICA,INC_07_02_2020-EX-10.12-Master Service Agreement",
     "Effective Date"): "5/25/08",                        # span: "25/05/2008"
    ("SPHERE3DCORP_06_24_2020-EX-10.12-CONSULTING AGREEMENT",
     "Expiration Date"): "5/31/21",                       # span: "expiring May 31st 2021"
    ("ALLISONTRANSMISSIONHOLDINGSINC_12_15_2014-EX-99.1-COOPERATION AGREEMENT",
     "Effective Date"): "12/12/14",                       # span: "December 12, 2014"
    ("KUBIENT,INC_07_02_2020-EX-10.14-MASTER SERVICES AGREEMENT_Part1",
     "Effective Date"): "2/5/20",                         # span: "the 5th day of February, 2020"
    # CUAD recorded a single renewal where the span renews indefinitely.
    ("NICELTD_06_26_2003-EX-4.5-OUTSOURCING AGREEMENT",
     "Renewal Term"): "perpetual",              # "automatic renewal for an indefinite period"
    ("NANOPHASETECHNOLOGIESCORP_11_01_2005-EX-99.1-DISTRIBUTOR AGREEMENT",
     "Renewal Term"): "successive 2 years",     # "thereafter automatically renew for additional two (2) year terms"
    ("BELLRINGBRANDS,INC_02_07_2020-EX-10.18-MASTER SUPPLY AGREEMENT",
     "Renewal Term"): "successive 5 years",     # "automatically renew for additional periods of five (5) years"
    ("ULTRAGENYXPHARMACEUTICALINC_12_23_2013-EX-10.9-SUPPLY AGREEMENT",
     "Renewal Term"): "successive 2 years",     # "automatically renewed for additional two year periods"
    ("MOELIS_CO_03_24_2014-EX-10.19-STRATEGIC ALLIANCE AGREEMENT",
     "Renewal Term"): "successive 1 year",  # "At the end of such initial term, and any renewed term"
    # Periods the span states plainly and the answer left empty.
    ("HYPERIONSOFTWARECORP_09_28_1994-EX-10.47-EXCLUSIVE DISTRIBUTOR AGREEMENT",
     "Notice Period to Terminate Renewal"): "90 days",   # "at least ninety (90) days prior to the end"
    ("LEGACYTECHNOLOGYHOLDINGS,INC_12_09_2005-EX-10.2-DISTRIBUTOR AGREEMENT",
     "Notice Period to Terminate Renewal"): "60 days",   # "sixty (60) days notice ... prior to the renewal date"
    ("BLUEFLYINC_03_27_2002-EX-10.27-e-business Hosting Agreement",
     "Notice Period to Terminate Renewal"): "90 days",   # "at least ninety (90) days prior to the end"
    # Renewal lengths the span states plainly and the answer left empty.
    ("PapaJohnsInternationalInc_20190617_8-K_EX-10.1_11707365_EX-10.1_Endorsement Agreement",
     "Renewal Term"): "1 year",            # "extended for one (1) year upon the parties' mutual agreement"
    ("TICKETSCOMINC_06_22_1999-EX-10.22-SPONSORSHIP AGREEMENT",
     "Renewal Term"): "1 year",            # "right to renew the Agreement for another year"
    ("NOVOINTEGRATEDSCIENCES,INC_12_23_2019-EX-10.1-JOINT VENTURE AGREEMENT",
     "Renewal Term"): "5 years",           # "a subsequent renewal of a five (5) year term"
    ("KINGPHARMACEUTICALSINC_08_09_2006-EX-10.1-PROMOTION AGREEMENT",
     "Renewal Term"): "successive 1 year", # "extended for subsequent one year periods upon mutual agreement"
    ("FOUNDATIONMEDICINE,INC_02_02_2015-EX-10.2-Collaboration Agreement",
     "Renewal Term"): "successive 3 years; 6 1 years",  # "additional three (3) year periods"; "up to six (6) additional one (1) year periods"
    # Contracts that renew but state no length for the renewal.
    ("CHIPMOSTECHNOLOGIESBERMUDALTD_04_18_2016-EX-4.72-Strategic Alliance Agreement",
     "Renewal Term"): "unspecified",       # "may negotiate for an extension"
    ("CHEETAHMOBILEINC_04_22_2014-EX-10.43-Cooperation Agreement",
     "Renewal Term"): "unspecified",       # "the parties may further negotiate the cooperation forms"
    ("IVILLAGEINC_03_17_1999-EX-10.16-SPONSORSHIP AGREEMENT",
     "Renewal Term"): "unspecified",       # renewal "on terms set forth in a proposal"
    ("BLUEFLYINC_03_27_2002-EX-10.27-e-business Hosting Agreement",
     "Renewal Term"): "unspecified",       # "an additional term equal in duration to the previous term"
    ("NEOMEDIATECHNOLOGIESINC_12_15_2005-EX-16.1-DISTRIBUTOR AGREEMENT",
     "Renewal Term"): "unspecified",       # CUAD's "30 days" is the notice window; no length is stated
    # A notice period the span states but the answer left empty, and one the contract never states.
    ("GOOSEHEADINSURANCE,INC_04_02_2018-EX-10.6-Franchise Agreement",
     "Notice Period to Terminate Renewal"): "180 days",   # the span quotes Minn. Stat. 80C.14: "180 days' notice of non-renewal"
    ("HEMISPHERX - Sales, Marketing, Distribution, and Supply Agreement",
     "Notice Period to Terminate Renewal"): "unspecified",  # "unless otherwise advised by one of the Parties"
    # Periods the filing blacks out ("[***]"), which CUAD left empty rather than marking redacted.
    ("AzulSa_20170303_F-1A_EX-10.3_9943903_EX-10.3_Maintenance Agreement1", "Renewal Term"): "[]",
    ("PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting Agreement", "Renewal Term"): "[]",
    ("KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement",
     "Renewal Term"): "[]",
    ("ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing Agreement",
     "Renewal Term"): "[]",
    ("RemarkHoldingsInc_20081114_10-Q_EX-10.24_2895649_EX-10.24_Content License Agreement",
     "Renewal Term"): "[]",
    ("BERKELEYLIGHTS,INC_06_26_2020-EX-10.12-COLLABORATION AGREEMENT",
     "Renewal Term"): "[] years",               # "extend for an additional [***] ([***]) year period"
    ("AzulSa_20170303_F-1A_EX-10.3_9943903_EX-10.3_Maintenance Agreement1",
     "Notice Period to Terminate Renewal"): "[]",
    ("FUSIONPHARMACEUTICALSINC_06_05_2020-EX-10.17-Supply Agreement - FUSION",
     "Notice Period to Terminate Renewal"): "[]",
    ("Magenta Therapeutics, Inc. - Master Development and Manufacturing Agreement",
     "Notice Period to Terminate Renewal"): "[]",
    ("PHREESIA,INC_05_28_2019-EX-10.18-STRATEGIC ALLIANCE AGREEMENT",
     "Notice Period to Terminate Renewal"): "[]",
    ("KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement",
     "Notice Period to Terminate Renewal"): "[]",
    ("ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing Agreement",
     "Notice Period to Terminate Renewal"): "[]",
    # Contracts filed with the effective date left blank.
    ("StaarSurgicalCompany_20180801_10-Q_EX-10.37_11289449_EX-10.37_Distributor Agreement",
     "Effective Date"): "[]/[]/[]",                       # span: "____________"
    ("JOINTCORP_09_19_2014-EX-10.15-FRANCHISE AGREEMENT",
     "Effective Date"): "[]/[]/20[]",                     # span: "_____ day of _______________, 20__"
    # Party lists their own spans contradict: parties merged into one entry, dropped, or repeated.
    ("VgrabCommunicationsInc_20200129_10-K_EX-10.33_11958828_EX-10.33_Development Agreement",
     "Parties"): 'VGrab Asia Ltd. ("VAL"); Mr. Zheng Qing; Mr. Gu Xianwin; Ms. Chen Weijie',  # all three developers sign
    ("HEALTHGATEDATACORP_11_24_1999-EX-10.1-HOSTING AND MANAGEMENT AGREEMENT - Escrow Agreement",
     "Parties"): 'NCC ESCROW INTERNATIONAL LIMITED ("NCC"); the Owner; the Licensee',     # three signature blocks
    ("BerkshireHillsBancorpInc_20120809_10-Q_EX-10.16_7708169_EX-10.16_Endorsement Agreement",
     "Parties"): 'Geno Auriemma ("Auriemma"); Berkshire Bank ("Berkshire")',            # Berkshire Bank was dropped
    ("OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY AGREEMENT2",
     "Parties"): 'Tribute Pharmaceuticals Inc. ("Corporation"); [] ("Agent"); [] ("U.S. Affiliate")',  # U.S. Affiliate listed twice
}

# Annotations kept as-is but reported, because reading the span suggests they are wrong.
FLAGS = {
    ("UsioInc_20040428_SB-2_EX-10.11_1723988_EX-10.11_Affiliate Agreement 2", "Governing Law"):
        "the spans name arbitration venues (McLean, Virginia; San Antonio, Texas), not a governing law",
    ("DRAGONSYSTEMSINC_01_08_1999-EX-10.17-OUTSOURCING AGREEMENT", "Governing Law"):
        "the span names court venues (Massachusetts or the Netherlands), not a governing law",
}

# Expirations whose term length is blacked out in the filing ("an initial term of ****"), or left
# blank ("through and until _______"), rather than tied to an event.
REDACTED_EXPIRATIONS = {
    "KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement",
    "ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing Agreement",
    "PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting Agreement",
    "IbioInc_20200313_8-K_EX-10.1_12052678_EX-10.1_Development Agreement",
    "NeuroboPharmaceuticalsInc_20190903_S-4_EX-10.36_11802165_EX-10.36_Manufacturing Agreement_ Supply Agreement",
    "MACROGENICSINC_08_02_2013-EX-10-COLLABORATION AGREEMENT",
    "SPORTHALEYINC_09_29_1997-EX-10.2-10-ENDORSEMENT AGREEMENT",
    "StaarSurgicalCompany_20180801_10-Q_EX-10.37_11289449_EX-10.37_Distributor Agreement",
}

# Expiration terms the span's own wording gives, where reading the first duration in the span gets
# it wrong. Checked one by one against the span text.
EXPIRATION_TERMS = {
    # "the Term shall be for one hundred eighty days (180) ... unless [approval fails], in which
    # case the Term will be 3 years"
    "UsioInc_20040428_SB-2_EX-10.11_1723988_EX-10.11_Affiliate Agreement 2": {"length": 180, "unit": "days"},
    # terms that end on an event; the duration elsewhere in the span is not the term
    "RareElementResourcesLtd_20171019_SC 13D_EX-99.4_10897534_EX-99.4_Intellectual Property Agreement":
        UNSPECIFIED,   # "until the expiration of the last to expire of the Patents"
    "BloomEnergyCorp_20180321_DRSA (on S-1)_EX-10_11240356_EX-10_Maintenance Agreement":
        UNSPECIFIED,   # first day to last day of the Warranty Period
    "ATHENSBANCSHARESCORP_11_02_2009-EX-1.2-AGENCY AGREEMENT , 2009":
        UNSPECIFIED,   # "upon termination of the Offering, but in no event later than 45 days after"
    "EbixInc_20010515_10-Q_EX-10.3_4049767_EX-10.3_Co-Branding Agreement":
        UNSPECIFIED,   # "expire upon delivery of [**] ... but in no way later than thirty (30) months"
    "ReedsInc_20191113_10-Q_EX-10.4_11888303_EX-10.4_Development Agreement":
        UNSPECIFIED,   # "the longer of the first anniversary ... or the duration of the [other] Agreement"
}

# Contracts whose effective date is an event, not a calendar date: execution or the last
# signature, a closing, an approval, conditions precedent, an acceptance. Their answer is
# ["relative"]; where annotators resolved the event to a date, that date is dropped, since the
# contract itself does not state one. The prompt tells the model to answer "relative" for these.
RELATIVE_EFFECTIVE_DATES = {
    "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
    "CerenceInc_20191002_8-K_EX-10.4_11827494_EX-10.4_Intellectual Property Agreement",
    "ChinaRealEstateInformationCorp_20090929_F-1_EX-10.32_4771615_EX-10.32_Content License Agreement",
    "CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement",
    "SouthernStarEnergyInc_20051202_SB-2A_EX-9_801890_EX-9_Affiliate Agreement",
    "MEDIWOUNDLTD_01_15_2014-EX-10.6-SUPPLY AGREEMENT",
    "RareElementResourcesLtd_20171019_SC 13D_EX-99.4_10897534_EX-99.4_Intellectual Property Agreement",
    "ATMOSENERGYCORP_11_22_2002-EX-10.17-TRANSPORTATION SERVICE AGREEMENT",
    "HUBEIMINKANGPHARMACEUTICALLTD_09_19_2006-EX-10.1-OUTSOURCING AGREEMENT",
    "FUSIONPHARMACEUTICALSINC_06_05_2020-EX-10.17-Supply Agreement - FUSION",
    "GpaqAcquisitionHoldingsInc_20200123_S-4A_EX-10.6_11951677_EX-10.6_License Agreement",
    "SalesforcecomInc_20171122_10-Q_EX-10.1_10961535_EX-10.1_Reseller Agreement",
    "AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE DEVELOPMENT AND HOSTING AGREEMENT",
    "HEMISPHERX - Sales, Marketing, Distribution, and Supply Agreement",
    "TRANSPHORM,INC_02_14_2020-EX-10.12(1)-JOINT VENTURE AGREEMENT",
    "VALENCETECHNOLOGYINC_02_14_2003-EX-10-JOINT VENTURE CONTRACT",
    "FOUNDATIONMEDICINE,INC_02_02_2015-EX-10.2-Collaboration Agreement",
    "BloomEnergyCorp_20180321_DRSA (on S-1)_EX-10_11240356_EX-10_Maintenance Agreement",
    "CHAPARRALRESOURCESINC_03_30_2000-EX-10.66-TRANSPORTATION CONTRACT",
    "GSITECHNOLOGYINC_11_16_2009-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT between SONY ELECTRONICS INC. and GSI TECHNOLOGY, INC.",
    "LejuHoldingsLtd_20140121_DRS (on F-1)_EX-10.26_8473102_EX-10.26_Content License Agreement1",
    "CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement",
    "KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement",
    "BLACKSTONEGSOLONG-SHORTCREDITINCOMEFUND_05_11_2020-EX-99.(K)(1)-SERVICE AGREEMENT",
    "QIWI_06_16_2017-EX-99.(D)(2)-COOPERATION AGREEMENT",
    "ElPolloLocoHoldingsInc_20200306_10-K_EX-10.16_12041700_EX-10.16_Development Agreement",
    "NlsPharmaceuticsLtd_20200228_F-1_EX-10.14_12029046_EX-10.14_Development Agreement",
    "GOOSEHEADINSURANCE,INC_04_02_2018-EX-10.6-Franchise Agreement",
    "OPTIMIZEDTRANSPORTATIONMANAGEMENT,INC_07_26_2000-EX-6.6-DISTRIBUTOR AGREEMENT",
    # annotators resolved the event to a date; the contract states none
    "AULAMERICANUNITTRUST_04_24_2020-EX-99.8.77-SERVICING AGREEMENT",
    "AgapeAtpCorp_20191202_10-KA_EX-10.1_11911128_EX-10.1_Supply Agreement",
    "QuantumGroupIncFl_20090120_8-K_EX-99.2_3672910_EX-99.2_Hosting Agreement",
    "VitalibisInc_20180316_8-K_EX-10.2_11100168_EX-10.2_Hosting Agreement",
    "GarrettMotionInc_20181001_8-K_EX-2.4_11364532_EX-2.4_Intellectual Property Agreement",
    "BLUEFLYINC_03_27_2002-EX-10.27-e-business Hosting Agreement",
    "IMPCOTECHNOLOGIESINC_04_15_2003-EX-10.65-JOINT VENTURE AGREEMENT",
}

US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
}
# Countries the raw answers name, so a "Region, Country" answer can be told from two US states
# ("Delaware, Illinois").
COUNTRIES = {
    "United States", "United Kingdom", "Canada", "China", "Germany", "Kazakhstan", "South Africa",
    "Israel", "Japan", "Netherlands", "India", "Switzerland", "Taiwan", "Italy", "Spain", "Colombia",
    "Papua New Guinea", "Singapore", "Belgium", "Australia", "Hong Kong",
}
# Sub-national places the raw answers name, outside the US.
SUBNATIONAL = {"England", "Wales", "Ontario", "British Columbia", "Nova Scotia", "Victoria", "Beijing"}
# Formal wrappers the raw answers put around a place name; the place itself is what is kept.
PLACE_PREFIX_RE = re.compile(
    r"^(the )?((people's|federal) )?(republic|commonwealth|state|province|city) of ", re.I)
LAW_TYPOS = {"Massachussets": "Massachusetts", "UK": "United Kingdom"}

WORD_NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
DURATION = r"(?P<length>[\d.]+|\[\])(?: ?(?P<unit>day|week|month|year)s?)?"
RENEWAL_RE = re.compile(r"(?:(?P<count>\d+) )?(?P<successive>successive )?" + DURATION)
NOTICE_RE = re.compile(DURATION)
MONTHS = [m[:3] for m in "jan feb mar apr may jun jul aug sep oct nov dec".split()]
WRITTEN_DATE_RE = re.compile(  # "September 7, 1999", "7th day of September, 1999"
    r"\b([a-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)?,? ?(\d{4})\b"
    r"|\b(\d{1,2})(?:st|nd|rd|th)? (?:day of )?([a-z]{3,9}),? ?(\d{4})\b")
LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|llc|l\.l\.c|ltd|limited|corp|corporation|company|co|l\.p|lp|plc|gmbh|ag|s\.a|sa|"
    r"b\.v|nv|pty|llp|trust|fund|bank|university|association)\b\.?", re.I)
# Wording that introduces another name for the same party, not a second party.
ANOTHER_NAME_RE = re.compile(
    r"formerly|known as|[fan]/k/a|aka|d/b/a|dba|doing business|on behalf|subsidiary of|successor", re.I)
COLLECTIVE_RE = re.compile(r"\b(collectively|individually|hereinafter|referred)\b", re.I)
DATE_RE = re.compile(r"\d{1,2}/\S+/\S+")


class Issues:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, category: str, code: str, detail: str = "") -> None:
        self.rows.append({"name": name, "category": category, "issue": code, "detail": detail})


def _file_key(filename: str) -> str:
    """Join key for a CUADv1 title and a master_clauses.csv Filename: the csv writes "&" and "'"
    where the json has "_", and a few csv names carry a stray ".PDF'" or trailing "-"."""
    stem = re.sub(r"(\.pdf'?)+$", "", filename.strip(), flags=re.I)
    return re.sub(r"[^A-Z0-9]", "", stem.upper())


def _split(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(";") if part.strip()]


def _number(text: str) -> int | float | str:
    if text == "[]":
        return REDACTED
    value = float(text)
    return int(value) if value.is_integer() else value


# -- dates ----------------------------------------------------------------------
def _year(raw: str) -> str:
    if re.fullmatch(r"\d{2}", raw):  # CUAD's m/d/yy: years 90-99 are 1990s, 00-45 are 2000s
        return ("19" if int(raw) >= 50 else "20") + raw
    return raw.replace("[]", "?").ljust(4, "?")


def _year_in_text(year: str, text: str) -> bool:
    """Whether a span writes this year, in full ("2014") or two-digit ("5/8/14", "3-31-16")."""
    return "?" in year or year in text or bool(re.search(rf"[/'’-]{year[2:]}\b", text))


def _written_dates(text: str) -> set[tuple[str, str, str]]:
    dates = set()
    for m in WRITTEN_DATE_RE.finditer(" ".join(text.lower().split())):
        month, day, year = (m[1], m[2], m[3]) if m[1] else (m[5], m[4], m[6])
        if month[:3] in MONTHS:
            dates.add((f"{MONTHS.index(month[:3]) + 1:02d}", day.zfill(2), year))
    return dates


def parse_dates(raw: str, name: str, category: str, issues: Issues) -> list:
    dates: list = []
    for part in _split(raw):
        if part.lower() in ("perpetual", RELATIVE):
            dates.append(part.lower())
            continue
        month, day, year = re.fullmatch(r"(\S+)/(\S+)/(\S+)", part.replace(" ", "")).groups()
        month, day = ("??" if "[]" in p else p.zfill(2) for p in (month, day))
        if month != "??" and int(month) > 12:
            month, day = day, month
            issues.add(name, category, "day_month_swapped", part)
        dates.append({"month": month, "day": day, "year": _year(year)})
    return dates


# -- durations --------------------------------------------------------------------
def _normalize_duration(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b[a-z]+ \((\d+)\)", r"\1", text)  # "three (3) years" -> "3 years"
    text = re.sub(r"[()]", "", text)                  # "(6) months", "60) days"
    text = re.sub(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", lambda m: WORD_NUMBERS[m[1]], text)
    text = re.sub(r"\bsuc+e?s+i[vn]e\b", "successive", text)  # succesive, sucessive, successine, ...
    return " ".join(text.split())


def _unit(match: re.Match) -> str:
    return match["unit"] + "s" if match["unit"] else REDACTED


def _split_durations(raw: str) -> list[str]:
    """Durations are listed with ";", but a few answers use "," ("4 years, 6 months") or "/"
    ("180 days / 6 months"). Dates keep their slashes."""
    parts = []
    for part in re.split(r"[;,]", raw):
        parts += [part] if DATE_RE.fullmatch(part.strip()) else part.split("/")
    return [p.strip() for p in parts if p.strip()]


def parse_renewal(raw: str, name: str, category: str, issues: Issues) -> list:
    terms: list = []
    for part in _split_durations(raw):
        if DATE_RE.fullmatch(part):  # a few answers give the renewal's end date instead
            terms += parse_dates(part, name, category, issues)
            continue
        part = _normalize_duration(part)
        if part in ("perpetual", REDACTED, UNSPECIFIED, "[]"):
            terms.append(REDACTED if part == "[]" else part)
            continue
        m = RENEWAL_RE.fullmatch(part)
        renewals = int(m["count"]) if m["count"] else ("unlimited" if m["successive"] else 1)
        terms.append({"renewals": renewals, "length": _number(m["length"]), "unit": _unit(m)})
    return terms


def parse_notice(raw: str) -> list:
    periods = []
    for part in _split_durations(raw):
        part = _normalize_duration(part)
        if part in ("[]", REDACTED, UNSPECIFIED):
            periods.append(REDACTED if part == "[]" else part)
            continue
        m = NOTICE_RE.fullmatch(part)
        periods.append({"length": _number(m["length"]), "unit": _unit(m)})
    return periods


# -- governing law ----------------------------------------------------------------
def _place(text: str) -> str:
    """The most specific place the answer names: "Beijing" from "Beijing", "Ontario" from
    "Ontario, Canada", "Kazakhstan" from "Republic of Kazakhstan". A model naming the country as
    well ("Beijing, China") still contains it."""
    text = PLACE_PREFIX_RE.sub("", text.strip()).strip()
    return LAW_TYPOS.get(text, text)


def parse_governing_law(raw: str, name: str, issues: Issues) -> list:
    places: list = []
    for part in _split(raw):
        if part.strip("[]* ") == "":
            places.append(REDACTED)
            continue
        pieces = [_place(p) for p in re.split(r",|\band\b", part) if p.strip()]
        if len(pieces) > 1 and pieces[-1] in COUNTRIES and not all(p in US_STATES for p in pieces):
            pieces = pieces[:-1]          # "Ontario, Canada" -> Ontario; "England and Wales" -> both
        for piece in pieces:
            if piece in US_STATES or piece in COUNTRIES or piece in SUBNATIONAL:
                places.append(piece)
            else:                          # "the state in which the breach occurs", a statute
                places.append(NON_PLACE)
                issues.add(name, "Governing Law", "not_a_jurisdiction", piece)
    return list(dict.fromkeys(places))


# -- parties ----------------------------------------------------------------------
def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _key(a), _key(b)).ratio()


def _respell(word: str, like: str) -> str:
    """`word`, taken from a span, wearing the case and trailing punctuation of the answer's `like`."""
    word = re.sub(r"^[^\w&]+|[^\w&]+$", "", word)
    if like.isupper():
        word = word.upper()
    elif word.isupper():  # an all-caps span, a Title Case answer
        word = word.capitalize()
    return word + re.search(r"[^\w&]*$", like)[0]


def _fix_typos(text: str, spans: list[str], contract_words: set[str]) -> str:
    """Replace the words of `text` that the contract never uses with their spelling in the Parties
    spans ("Mount Kowledge" -> "Mount Knowledge", "Phototronics" -> "Photronics").

    A word the contract does use is never touched, so the defined term "Marathon Parties" is not
    "corrected" to the span "Marathon Partners". Two passes: first word-for-word against the one
    span `text` resembles, which can correct even a short word from its position ("LCC" -> "LLC");
    then, for words still unknown to the contract, against the span vocabulary as a whole, which
    reaches the words of an entry that names two parties at once ("Phototronics and DNP")."""
    if not spans or all(_key(w) in contract_words for w in text.split()):
        return text
    words = text.split()

    best = max(spans, key=lambda s: _similarity(text, s))
    if _similarity(text, best) >= 0.8:
        span_words = best.split()
        matcher = difflib.SequenceMatcher(None, [_key(w) for w in words], [_key(w) for w in span_words])
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op != "replace" or i2 - i1 != j2 - j1:
                continue
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if _key(words[i]) not in contract_words and _similarity(words[i], span_words[j]) >= 0.5:
                    words[i] = _respell(span_words[j], words[i])

    vocabulary = {w for s in spans for w in s.split() if len(_key(w)) >= 5}
    for i, word in enumerate(words):
        if len(_key(word)) < 5 or _key(word) in contract_words:
            continue
        match = max(vocabulary, key=lambda w: _similarity(word, w), default=None)
        # A typo is a word of the same shape, misspelled. One word containing the other is
        # something else -- a run-on ("VerticalNet,Inc"), a truncation ("Concepts" vs the
        # OCR-broken span "Concep"), a longer name -- and is left alone.
        if match and _similarity(word, match) >= 0.8 and abs(len(_key(word)) - len(_key(match))) <= 3 \
                and _key(word) not in _key(match) and _key(match) not in _key(word):
            words[i] = _respell(match, word)
    return " ".join(words)


def _in_spans(text: str, spans: list[str]) -> bool:
    return any(_key(text) in _key(s) or _similarity(text, s) >= 0.85 for s in spans)


def _count_is_unclear(parties: list[dict], spans: list[str]) -> str:
    """Whether an entry packs several parties into one, so the count cannot be read off the answer.

    Either it names two companies at once ("WYZZ, Inc. and WYZZ Licensee, Inc.") or it is the
    contract's collective for parties listed separately ("Phototronics and DNP", the
    "Shareholders"). The entry stays; only the count is reported as unreadable."""
    for party in parties:
        # Two legal suffixes with a name's worth of words between them: "WYZZ, Inc. and WYZZ
        # Licensee, Inc.". Adjacent suffixes are one company: "Nanjing Tuniu Technology Co., Ltd."
        if any(_key(party["name"]) == _key(s) for s in spans):  # one span, so one party
            continue
        suffixes = list(LEGAL_SUFFIX_RE.finditer(party["name"]))
        for before, after in zip(suffixes, suffixes[1:]):
            between = party["name"][before.end():after.start()]
            if (len(between.split()) >= 3 and re.search(r"\band\b|&|,|/", between)
                    and not ANOTHER_NAME_RE.search(between)):
                return f'names more than one party: {party["name"]}'
        # The contract's collective for parties the answer also lists separately: "Phototronics
        # and DNP", the "Shareholders".
        others = {_key(t) for p in parties if p is not party for t in [p["name"], *p["aliases"]]}
        pieces = [p.strip() for p in re.split(r"\band\b|&", party["name"]) if p.strip()]
        if len(pieces) >= 2 and all(_key(p) in others for p in pieces):
            return f'collective for parties listed separately: {party["name"]}'
    return ""


def parse_parties(raw: str, spans: list[str], contract: str, name: str, issues: Issues) -> dict:
    spans = [" ".join(s.split()) for s in spans]
    contract_words = {_key(w) for w in contract.split()}
    parties = []
    for entry in _split(raw):
        aliases, party = [], entry
        for group in re.findall(r"\(([^()]*)\)", entry):
            if re.search(r"[\"“”]", group):  # ("Company", "MA") holds aliases; (Pty) is part of a name
                aliases += re.findall(r"[\"“]([^\"“”]+)[\"”]", group)
                party = party.replace(f"({group})", " ")
        aliases += re.findall(r"[\"“]([^\"“”]+)[\"”]", party)  # 'Excite, Inc."Excite"'
        party = re.sub(r"[\"“][^\"“”]+[\"”]", " ", party)
        party = " ".join(party.split()).strip(" ,;")
        if party.count("(") > party.count(")"):
            party = party.replace("(", "", 1).strip()
        if party.count(")") > party.count("("):
            party = party.rstrip(") ").strip()
        aliases = [a.strip(" ,.") for a in aliases]
        is_party_word = [re.fullmatch(r"(the |a )?part(y|ies)", a.lower()) is not None for a in aliases]
        # The collective entries: 'Vericel and MediWound (individually as a "Party" and collectively
        # as the "Parties")', 'Each of the foregoing parties is referred to herein as a "Party" ...'
        if (not set(party.lower().split()) - {"and", "or", "together", "with", "the"}
                or (aliases and all(is_party_word)) or COLLECTIVE_RE.search(party)):
            continue
        aliases = list(dict.fromkeys(a for a, generic in zip(aliases, is_party_word) if a and not generic))

        fixed = _fix_typos(party, spans, contract_words)
        fixed_aliases = [_fix_typos(a, spans, contract_words) for a in aliases]
        for before, after in [(party, fixed), *zip(aliases, fixed_aliases)]:
            if before != after:
                issues.add(name, "Parties", "typo_fixed", f"{before} -> {after}")
        if not _in_spans(fixed, spans):
            issues.add(name, "Parties", "party_not_in_spans", fixed)
        parties.append({"name": fixed, "aliases": list(dict.fromkeys(fixed_aliases))})

    unclear = _count_is_unclear(parties, spans)
    if unclear:
        issues.add(name, "Parties", "number_parties_unclear", unclear)
    # Only the count and the names are kept. CUAD's names carry the annotators' own wording, so
    # grade a model's names against the Parties spans instead of against this list; the defined
    # terms ("Company", "MA") are in those spans too, and are not repeated here.
    return {"number_parties": None if unclear else len(parties), "parties": [p["name"] for p in parties]}


# -- assembly ---------------------------------------------------------------------
def _categories() -> list[str]:
    with CATEGORY_DESCRIPTIONS_CSV.open(encoding="utf-8-sig") as handle:
        return [row[0].removeprefix("Category: ") for row in list(csv.reader(handle))[1:]]


def _csv_answer(row: pd.Series, category: str) -> str | None:
    column = next(c for c in row.index if re.fullmatch(re.escape(category) + r"\s*-\s*Answer", c, re.I))
    value = row[column]
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_answer(category: str, raw: str, spans: list[str], contract: str, name: str, issues: Issues):
    if category == "Document Name":
        return " ".join(raw.split())
    if category == "Parties":
        return parse_parties(raw, spans, contract, name, issues)
    if category in DATE_CATEGORIES:
        return parse_dates(raw, name, category, issues)
    if category == "Renewal Term":
        return parse_renewal(raw, name, category, issues)
    if category == "Notice Period to Terminate Renewal":
        return parse_notice(raw)
    return parse_governing_law(raw, name, issues)


def _check_answer(category: str, answer, spans: list[str], name: str, issues: Issues) -> None:
    if category in ("Document Name", "Parties"):
        return
    items = answer if isinstance(answer, list) else [answer]
    if len(items) > 1:
        issues.add(name, category, "multiple_values", json.dumps(answer))
        lengths = {_length_in_days(i) for i in items if isinstance(i, dict) and "length" in i}
        dates = {json.dumps(i) for i in items if isinstance(i, dict) and "year" in i}
        # Two ways of writing one period ("180 days / 6 months") are both right, and a renewal may
        # genuinely run 5 years then 2; but one notice period or one date cannot be two values.
        if len(dates) > 1 or (category != "Renewal Term" and len(lengths - {None}) > 1):
            issues.add(name, category, "different_values", json.dumps(answer))
    if category in DATE_CATEGORIES and spans:
        span_text = " ".join(spans)
        for item in items:
            if not isinstance(item, dict) or "year" not in item or "?" in item["year"]:
                continue
            if not _year_in_text(item["year"], span_text):
                issues.add(name, category, "computed_date", json.dumps(item))
        # Expiration spans usually write the start date, from which the answer is computed.
        known = {(d["month"], d["day"], d["year"]) for d in items
                 if isinstance(d, dict) and "year" in d and "?" not in str(d)}
        written = _written_dates(span_text)
        if category != "Expiration Date" and known and written and not known & written:
            issues.add(name, category, "date_differs_from_span", f"answer {sorted(known)}, spans {sorted(written)}")


TERM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "fifteenth": 15,
    "twentieth": 20, "thirtieth": 30,
}
_NUM = r"(?:(\d+)|([a-z]+(?:[- ][a-z]+)?))\s*(?:\((\d+)(?:st|nd|rd|th)?\))?"
_UNIT = r"(year|month|day|week|contract year|anniversary)"
_CUE = r"(?:terms?|periods?|continu\w*|remain\w*|expir\w*|terminat\w*|extend\w*|valid|force|effect|duration|anniversar\w*|thereafter)"
TERM_RE = re.compile(rf"{_CUE}\b[^.]{{0,70}}?\b{_NUM}\s*[- ]?(?:calendar|consecutive|full|successive|contract|additional|initial)?\s*{_UNIT}s?", re.I)
TERM_NOTICE_RE = re.compile(rf"{_NUM}\s*[- ]?(?:calendar|consecutive|full)?\s*{_UNIT}s?[^.]{{0,25}}(?:prior|before|notice|advance)", re.I)


def _term_number(digits: str, word: str, parenthesised: str) -> int | None:
    if parenthesised or digits:
        return int(parenthesised or digits)
    words = (word or "").lower().replace("-", " ").split()
    if len(words) == 2 and all(w in TERM_WORDS for w in words):
        return TERM_WORDS[words[0]] + TERM_WORDS[words[1]]
    return TERM_WORDS.get(words[-1]) if words else None


def _term_in_spans(spans: list[str]) -> dict | None:
    """The term length the spans state: "a term of ten years", "the fifth (5th) anniversary of the
    Effective Date". A length that a notice period owns ("90 days prior to") is skipped."""
    text = " ".join(" ".join(s.split()) for s in spans)
    notices = {m.group(1, 2, 3) for m in TERM_NOTICE_RE.finditer(text)}
    for m in TERM_RE.finditer(text):
        if m.group(1, 2, 3) in notices:
            continue
        length = _term_number(*m.group(1, 2, 3))
        if not length:
            continue
        unit = m[4].lower()
        return {"length": length, "unit": ("year" if unit in ("anniversary", "contract year") else unit) + "s"}
    return None


def _term_between(start: dict | None, end: dict | None) -> dict | None:
    """The term CUAD's own computed expiration date implies, in whole months from the start date."""
    if not start or not end or any("?" in d[k] for d in (start, end) for k in ("month", "year")):
        return None
    months = (int(end["year"]) - int(start["year"])) * 12 + int(end["month"]) - int(start["month"])
    if not 0 < months <= 600:
        return None
    return {"length": months // 12, "unit": "years"} if months % 12 == 0 else {"length": months, "unit": "months"}


def _length_in_days(item: dict) -> int | None:
    """A duration in days, so "6 months" and "180 days" compare equal."""
    scale = {"years": 365, "months": 30, "weeks": 7, "days": 1}.get(item.get("unit"))
    return scale * item["length"] if scale and isinstance(item.get("length"), (int, float)) else None


def _months(term: dict) -> int | None:
    return term["length"] * 12 if term["unit"] == "years" else term["length"] if term["unit"] == "months" else None


def _relative_expiration(answer: list, spans: list[str], start: dict | None, name: str,
                         issues: Issues) -> list:
    """An expiration date the contract does not state -- because the annotators added the initial
    term to the start date, or because the term ends on an event -- becomes the term itself
    ({"length": 5, "unit": "years"}), never a computed date. Two independent readings have to
    agree: the length the span writes, and the length CUAD's own computed date implies."""
    out: list = []
    for item in answer:
        if item == RELATIVE:  # CUAD gave no date at all: the term is whatever the span states
            term = EXPIRATION_TERMS.get(name) or _term_in_spans(spans) \
                or (REDACTED if name in REDACTED_EXPIRATIONS else UNSPECIFIED)
            if term not in out:
                out.append(term)
            continue
        if not isinstance(item, dict) or _year_in_text(item["year"], " ".join(spans)):
            if item not in out:
                out.append(item)
            continue
        issues.add(name, "Expiration Date", "relative_expiration", json.dumps(item))
        written, implied = EXPIRATION_TERMS.get(name) or _term_in_spans(spans), _term_between(start, item)
        if isinstance(written, str):  # a checked "unspecified" wins over CUAD's computed date
            out.append(written)
            continue
        if written and implied and _months(written) is not None and abs(_months(written) - _months(implied)) > 1:
            issues.add(name, "Expiration Date", "term_unclear",
                       f"the span says {written}, CUAD's date implies {implied}")
            term = None
        else:
            term = written or implied
        if term is None:
            term = REDACTED if name in REDACTED_EXPIRATIONS else UNSPECIFIED
        if term not in out:
            out.append(term)
    return out


def build_row(doc: dict, csv_row: pd.Series, categories: list[str], issues: Issues) -> dict:
    name, contract = doc["title"], doc["paragraphs"][0]["context"]
    qas = {qa["id"].split("__", 1)[1].lower(): qa["answers"] for qa in doc["paragraphs"][0]["qas"]}
    row = {"name": name}
    for category in categories:
        spans = [a["text"] for a in sorted(qas[category.lower()], key=lambda a: a["answer_start"])]
        row[category] = json.dumps(spans, ensure_ascii=False)
        raw = _csv_answer(csv_row, category)

        if category not in ANSWER_CATEGORIES:
            answer = "Yes" if spans else "No"
            if raw and raw != answer:
                issues.add(name, category, "yes_no_differs_from_csv", f"csv {raw}, json {answer}")
            row[f"{category} Answer"] = answer
            continue

        if category == "Effective Date" and name in RELATIVE_EFFECTIVE_DATES:
            raw = RELATIVE
        if category == "Expiration Date" and spans and not raw:  # a term ending on an event
            raw = RELATIVE
        if (name, category) in OVERRIDES:
            issues.add(name, category, "override", f"{raw} -> {OVERRIDES[(name, category)]}")
            raw = OVERRIDES[(name, category)]
        if (name, category) in FLAGS:
            issues.add(name, category, "questionable", FLAGS[(name, category)])
        if raw and not spans:
            issues.add(name, category, "answer_without_span", raw)
        if spans and not raw:
            issues.add(name, category, "span_without_answer")

        if raw:
            answer = _parse_answer(category, raw, spans, contract, name, issues)
        else:  # no answer: an empty value of the shape this category's answers take
            answer = {"Document Name": None, "Parties": {"number_parties": None, "parties": []}}.get(category, [])
        if category == "Expiration Date" and spans:
            start = next((d for c in ("Effective Date", "Agreement Date")
                          for d in json.loads(row[f"{c} Answer"]) if isinstance(d, dict)), None)
            answer = _relative_expiration(answer, spans, start, name, issues)
        if raw:
            _check_answer(category, answer, spans, name, issues)
        row[f"{category} Answer"] = json.dumps(answer, ensure_ascii=False)
    return row


def _unusable_column(name: str, issues: Issues) -> dict:
    reasons = {}
    for row in issues.rows:
        if row["name"] == name and row["issue"] in UNUSABLE_ISSUES:
            key = UNUSABLE_KEYS.get(row["issue"], row["category"])
            reasons[key] = UNUSABLE_ISSUES[row["issue"]]
    for (title, category), reason in UNUSABLE.items():
        if title == name:
            reasons[category] = reason
    return reasons


def main() -> None:
    documents = json.loads(CUAD_JSON.read_text())["data"]
    master = pd.read_csv(cuad_ground_truth_csv())
    master_by_key = {_file_key(f): row for f, (_, row) in zip(master["Filename"], master.iterrows())}
    categories = _categories()
    missing = [d["title"] for d in documents if _file_key(d["title"]) not in master_by_key]
    if missing:
        raise KeyError(f"no master_clauses.csv row for {missing}")

    named = set(EXPIRATION_TERMS) | REDACTED_EXPIRATIONS | {t for t, _ in UNUSABLE} | {t for t, _ in OVERRIDES} | {t for t, _ in FLAGS} | RELATIVE_EFFECTIVE_DATES
    unknown = named - {d["title"] for d in documents}
    if unknown:
        raise KeyError(f"OVERRIDES/FLAGS/RELATIVE_EFFECTIVE_DATES name no such contract: {sorted(unknown)}")

    issues = Issues()
    rows = [build_row(d, master_by_key[_file_key(d["title"])], categories, issues) for d in documents]
    for row in rows:  # after every issue for the contract has been collected
        row["unusable"] = json.dumps(_unusable_column(row["name"], issues), ensure_ascii=False)

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    issue_df = pd.DataFrame(issues.rows, columns=["name", "category", "issue", "detail"])
    issue_df.to_csv(ISSUES_CSV, index=False)
    print(f"wrote {len(rows)} contracts x {len(categories)} categories -> {OUTPUT_CSV}")
    print(f"wrote {len(issue_df)} issues -> {ISSUES_CSV}")
    print(issue_df.groupby(["category", "issue"]).size().to_string())


if __name__ == "__main__":
    main()
