# 🔍 Drug Discovery Assistant - Find Compounds You Didn't Know Existed!

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## The Problem This Solves

**"What if she needs compounds she doesn't know exist because she hasn't seen these ingredients?"**

This tool helps you **discover** the 68+ drugs in the database through multiple exploration methods!

---

## 🎯 Quick Start

```bash
# Interactive discovery mode (recommended!)
python drug_discovery_assistant.py

# Quick searches
python drug_discovery_assistant.py cancer breast     # Drugs for breast cancer
python drug_discovery_assistant.py natural           # All natural compounds
python drug_discovery_assistant.py repurposed        # Off-label drugs
python drug_discovery_assistant.py low-tox           # Minimal side effects
python drug_discovery_assistant.py all               # Browse everything
```

---

## 🔎 Discovery Methods

### 1. By Cancer Type
Find drugs you didn't know treated your specific cancer:

```python
assistant.discover_by_cancer_type('breast')
# Returns: Trastuzumab, Doxorubicin, Paclitaxel, Tamoxifen, etc.

assistant.discover_by_cancer_type('melanoma')
# Returns: Vemurafenib, Dabrafenib, Pembrolizumab, Ipilimumab, etc.
```

**Supported types:**
- breast, lung, colorectal, prostate, melanoma
- pancreatic, glioblastoma, ovarian, leukemia, lymphoma

---

### 2. By Mechanism of Action
Discover drugs by how they work:

```python
assistant.discover_by_mechanism('DNA')
# Returns: All DNA-damaging agents (cisplatin, doxorubicin, bleomycin, etc.)

assistant.discover_by_mechanism('kinase')
# Returns: All kinase inhibitors (erlotinib, imatinib, dabrafenib, etc.)

assistant.discover_by_mechanism('checkpoint')
# Returns: All checkpoint inhibitors (pembrolizumab, nivolumab, ipilimumab)
```

**Common mechanisms:**
- DNA, kinase, checkpoint, microtubule
- AMPK, apoptosis, angiogenic, autophagy
- oxidative, metabolic, epigenetic

---

### 3. Natural Compounds Only
Find all plant-based and supplement options:

```python
assistant.discover_natural_alternatives()
```

**Discovers 12+ natural compounds:**
- **Vitamins:** D3, C
- **Polyphenols:** Curcumin, Quercetin, Resveratrol, EGCG
- **Antimalarials:** Artemisinin
- **Alkaloids:** Berberine
- **Cannabinoids:** CBD
- **Hormones:** Melatonin
- **Fatty acids:** Omega-3 DHA
- **Phytochemicals:** Sulforaphane

Each with source info (where to get it naturally)!

---

### 4. Repurposed Drugs
FDA-approved drugs used off-label for cancer:

```python
assistant.discover_repurposed_drugs()
```

**Finds:**
- **Metformin** (diabetes → cancer metabolism)
- **Ivermectin** (antiparasitic → PAK1 inhibitor)
- **Mebendazole** (antiparasitic → tubulin inhibitor)
- **Hydroxychloroquine** (malaria → autophagy inhibitor)
- **Aspirin** (pain → COX-2 inhibitor)
- **Dichloroacetate** (lactic acidosis → Warburg reversal)
- **Fenbendazole** (veterinary → tubulin + GLUT inhibitor)

---

### 5. By Molecular Target
Find all drugs hitting a specific protein:

```python
assistant.discover_by_target('EGFR')
# Returns: Erlotinib, Osimertinib, Lapatinib, EGCG, etc.

assistant.discover_by_target('BRAF')
# Returns: Vemurafenib, Dabrafenib, Sorafenib

assistant.discover_by_target('PD-1')
# Returns: Pembrolizumab, Nivolumab
```

**Common targets:**
- EGFR, BRAF, ALK, ROS1, MET, KIT
- PD-1, PD-L1, CTLA-4
- VEGF, HER2, BCR-ABL
- ER, AR, DNA

---

### 6. Low Toxicity Options
Find gentle drugs with minimal side effects:

```python
assistant.discover_low_toxicity()
```

**Returns drugs with toxicity score < 0.5:**
- Most natural compounds (vitamins, polyphenols)
- Some targeted therapies (imatinib, crizotinib)
- Hormone therapies (tamoxifen, letrozole)
- Metabolic inhibitors (metformin, berberine)

---

### 7. By Drug Class
Explore entire categories:

```python
assistant.discover_by_class('immunotherapy')
# Returns: All 5 checkpoint inhibitors

assistant.discover_by_class('hormone')
# Returns: All 5 hormone therapies

assistant.discover_by_class('metabolic')
# Returns: All 19 metabolic/natural compounds
```

**Available classes:**
- chemotherapy (23 drugs)
- targeted_therapy (15 drugs)
- immunotherapy (5 drugs)
- hormone_therapy (5 drugs)
- metabolic_inhibitor (19 drugs)
- antiangiogenic (1 drug)

---

### 8. Combination Suggestions
Get smart recommendations based on what you're already using:

```python
assistant.discover_novel_combinations(['cisplatin', 'paclitaxel'])
```

**Returns:**
- **Synergistic:** Drugs that work well with most things (immunotherapy)
- **Different mechanisms:** Complementary pathways
- **Metabolic support:** Enhance other drugs (metformin, DCA, berberine)

---

## 💡 Interactive Mode Examples

### Example 1: "I have breast cancer, what can I use?"

```
$ python drug_discovery_assistant.py
Select option: 1 (By cancer type)
Enter cancer type: breast

✓ Found 8 drugs:
  • Trastuzumab (HER2 antibody)
  • Doxorubicin (anthracycline chemotherapy)
  • Paclitaxel (microtubule stabilizer)
  • Tamoxifen (SERM for ER+ breast)
  • Letrozole (aromatase inhibitor)
  • Lapatinib (dual EGFR/HER2 TKI)
  • etc.
```

---

### Example 2: "Show me only natural options"

```
Select option: 3 (Natural compounds)

✓ Found 12 natural compounds:
  • Vitamin D3 - Source: Sunlight, fish oil
  • Curcumin - Source: Turmeric root
  • Quercetin - Source: Onions, apples, berries
  • Artemisinin - Source: Sweet wormwood
  • CBD - Source: Cannabis/hemp
  • etc.
```

---

### Example 3: "I want something that targets EGFR"

```
Select option: 5 (By molecular target)
Enter target: EGFR

✓ Found 4 drugs:
  • Erlotinib (1st-gen EGFR TKI)
  • Osimertinib (3rd-gen, targets T790M)
  • Lapatinib (dual EGFR/HER2)
  • EGCG (natural EGFR inhibitor from green tea)
```

---

### Example 4: "What goes well with cisplatin?"

```
Select option: 9 (Suggest combinations)
Enter drugs: cisplatin

SUGGESTED COMBINATIONS:
✓ Synergistic:
  • Pembrolizumab (checkpoint inhibitor)
  • Nivolumab (checkpoint inhibitor)

✓ Different mechanisms:
  • Imatinib (tyrosine kinase inhibitor)
  • Vemurafenib (BRAF inhibitor)

✓ Metabolic support:
  • Metformin (AMPK activator)
  • Dichloroacetate (Warburg reversal)
  • Berberine (mitochondrial targeting)
```

---

## 📊 What You'll Discover

### Complete Arsenal: **68 Drugs**

| Category | Count | What You'll Find |
|----------|-------|------------------|
| **Chemotherapy** | 23 | Platinum agents, taxanes, antimetabolites, vinca alkaloids, anthracyclines, etc. |
| **Targeted Therapy** | 15 | EGFR/BRAF/ALK inhibitors, PARP inhibitors, MEK inhibitors, antibodies |
| **Immunotherapy** | 5 | PD-1, PD-L1, CTLA-4 checkpoint inhibitors |
| **Hormone Therapy** | 5 | SERMs, aromatase inhibitors, AR antagonists |
| **Metabolic/Natural** | 19 | Vitamins, polyphenols, repurposed drugs, experimental compounds |
| **Antiangiogenic** | 1 | VEGF inhibitor |

---

## 🎓 How to Use This for Research

### Discovery Workflow:

1. **Start broad** - "Show me all drugs for my cancer type"
2. **Learn mechanisms** - "What do these drugs actually do?"
3. **Find natural alternatives** - "Are there gentler options?"
4. **Check repurposed** - "What off-label drugs exist?"
5. **Find combinations** - "What works well together?"
6. **Filter by toxicity** - "Show me only low side-effect options"

### Example Research Session:

```bash
# Session 1: Discovery
python drug_discovery_assistant.py cancer lung      # Find lung cancer drugs
python drug_discovery_assistant.py natural          # See natural alternatives

# Session 2: Deep dive
python drug_discovery_assistant.py                  # Interactive mode
# Select "By mechanism" → Search for "kinase"
# Select "By target" → Search for "EGFR"

# Session 3: Build protocol
# Now you know: Erlotinib (EGFR), Osimertinib (EGFR T790M),
#               Curcumin (natural), Metformin (metabolic support)

# Session 4: Test in lab
lab.administer_drug("erlotinib", 150.0)
lab.administer_drug("curcumin", 1000.0)
lab.administer_drug("metformin", 1000.0)
lab.run_experiment(duration_days=30)
```

---

## 🔬 Integration with Oncology Lab

Once you discover drugs, test them:

```python
from oncology_lab import OncologyLaboratory, OncologyLabConfig, TumorType, CancerStage
from drug_discovery_assistant import DrugDiscoveryAssistant

# Step 1: Discover
assistant = DrugDiscoveryAssistant()
drugs = assistant.discover_by_cancer_type('breast')

# Step 2: Select interesting ones
print("Discovered drugs:")
for drug in drugs[:5]:
    print(f"  • {drug['name']} - {drug['mechanism']}")

# Step 3: Test in lab
lab = OncologyLaboratory(OncologyLabConfig(
    tumor_type=TumorType.BREAST_CANCER,
    stage=CancerStage.STAGE_II,
))

lab.administer_drug('trastuzumab', 420.0)
lab.administer_drug('paclitaxel', 175.0)
lab.administer_drug('curcumin', 1000.0)  # Natural addition

lab.run_experiment(duration_days=21)
results = lab.get_results()
```

---

## ⚡ Command-Line Quick Reference

```bash
# Interactive (best for exploration)
python drug_discovery_assistant.py

# Direct queries (fast)
python drug_discovery_assistant.py cancer <type>
python drug_discovery_assistant.py natural
python drug_discovery_assistant.py repurposed
python drug_discovery_assistant.py low-tox
python drug_discovery_assistant.py all
```

---

## 🎯 Key Benefits

### 1. **Serendipitous Discovery**
Find drugs you'd never think to search for because you didn't know they existed!

### 2. **Cross-Domain Learning**
Discover that antimalarial drugs (artemisinin) or antiparasitics (ivermectin) have anti-cancer properties.

### 3. **Natural Options**
Learn about evidence-based natural compounds you can obtain easily.

### 4. **Mechanism Understanding**
Group drugs by what they actually do, not just their names.

### 5. **Smart Combinations**
Get AI-assisted suggestions for synergistic protocols.

---

## 📈 Database Coverage

**68 drugs spanning:**
- ✅ 1960s-2020s (complete historical coverage)
- ✅ All major drug classes
- ✅ FDA-approved + experimental
- ✅ Conventional + natural
- ✅ Common + rare cancers
- ✅ All molecular targets (EGFR, BRAF, ALK, PD-1, etc.)

**Each drug includes:**
- ✅ Real PK/PD parameters from literature
- ✅ Mechanism of action
- ✅ Molecular targets
- ✅ FDA approval status & year
- ✅ Approved indications
- ✅ Toxicity profiles

---

## 🌟 Example Discoveries

### "I didn't know ivermectin was studied for cancer!"
```
assistant.discover_repurposed_drugs()
→ Ivermectin: PAK1/Akt/mTOR inhibitor, anti-mitotic
   Original use: Antiparasitic
   Cancer studies: Multiple in vitro studies showing efficacy
```

### "There's a green tea compound that inhibits EGFR?"
```
assistant.discover_by_target('EGFR')
→ EGCG: Natural EGFR inhibitor from green tea
   Source: 3-4 cups of green tea daily
   Also targets VEGFR (anti-angiogenic)
```

### "What works with checkpoint inhibitors?"
```
assistant.discover_novel_combinations(['pembrolizumab'])
→ Suggestions:
   • Vemurafenib (BRAF inhibitor) - synergistic in melanoma
   • Dabrafenib + Cobimetinib - triple combo
   • Metformin - metabolic support
```

---

## 🎁 No More Missing Compounds!

With this discovery tool, you'll **never miss a potentially useful drug** because:

1. ✅ Browse by your specific cancer type
2. ✅ Filter by natural/repurposed/low-toxicity
3. ✅ Search by mechanism you're interested in
4. ✅ Find drugs targeting specific proteins
5. ✅ Get smart combination suggestions
6. ✅ Explore all 68 drugs interactively

**The database becomes explorable, not just searchable!**

---

**Last Updated:** November 2025  
**Drug Database:** 68 drugs  
**Discovery Methods:** 9 different exploration paths  
**Status:** Ready to discover! 🔍
