"""Generate a sample PDF about Renewable Energy Systems for demo and evaluation.

This script creates a 4-page PDF with rich content about solar, wind, and
energy storage technologies. The content is designed to support the Q&A pairs
in eval/qa_pairs.json.
"""

from fpdf import FPDF
import os


def generate_sample_pdf(output_path: str = "data/sample.pdf") -> None:
    """Generate the sample PDF document."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Page 1: Title + Solar Energy ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Renewable Energy Systems", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 14)
    pdf.cell(0, 10, "A Comprehensive Technical Overview", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 10, "2024 Edition", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Solar Photovoltaic Technology", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    solar_text = (
        "Solar photovoltaic (PV) technology converts sunlight directly into electricity "
        "using semiconductor materials. The most common material is crystalline silicon, "
        "which accounts for over 90% of the global PV market. When photons from sunlight "
        "strike the silicon cell, they knock electrons free from their atoms, creating an "
        "electric current through the photovoltaic effect.\n\n"
        "Modern commercial solar panels typically achieve conversion efficiencies between "
        "18% and 22%, with premium panels reaching up to 24%. Research laboratories have "
        "demonstrated efficiencies exceeding 47% using multi-junction cells, though these "
        "remain too expensive for mass deployment.\n\n"
        "The global installed solar capacity surpassed 1.2 TW in 2023, with China, the "
        "United States, and India leading in new installations. The levelized cost of "
        "solar energy (LCOE) has dropped by approximately 89% since 2010, making it the "
        "cheapest source of new electricity generation in most regions of the world.\n\n"
        "Solar energy produces no direct greenhouse gas emissions during operation, unlike "
        "fossil fuels which release CO2 and other pollutants. A typical residential solar "
        "installation offsets approximately 3-4 tonnes of CO2 annually, contributing "
        "significantly to climate change mitigation efforts."
    )
    pdf.multi_cell(0, 6, solar_text)

    # --- Page 2: Wind Energy ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Wind Energy Technology", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    wind_text = (
        "Wind energy harnesses the kinetic energy of moving air to generate electricity. "
        "Modern wind turbines can have rotor diameters exceeding 150 meters and generate "
        "up to 15 MW of power. Offshore wind farms benefit from stronger, more consistent "
        "wind patterns compared to onshore installations.\n\n"
        "The physics of wind power follows the Betz limit, which states that no wind "
        "turbine can capture more than 59.3% of the kinetic energy in the wind. Modern "
        "turbines achieve approximately 35-45% efficiency, approaching the theoretical "
        "maximum when accounting for mechanical and electrical losses.\n\n"
        "Global wind power capacity reached 906 GW by the end of 2023. The offshore wind "
        "sector has seen particularly rapid growth, with floating wind platforms enabling "
        "deployment in deeper waters where fixed-bottom foundations are impractical.\n\n"
        "Wind farm development requires careful site assessment including wind resource "
        "mapping, environmental impact studies, and grid connection planning. Modern wind "
        "farms use sophisticated SCADA (Supervisory Control and Data Acquisition) systems "
        "for real-time monitoring and optimization of turbine performance.\n\n"
        "The integration of wind energy into electrical grids presents challenges related "
        "to variability and forecasting. Advanced weather prediction models and grid "
        "management software help operators balance supply and demand, while energy "
        "storage systems provide additional flexibility."
    )
    pdf.multi_cell(0, 6, wind_text)

    # --- Page 3: Energy Storage ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. Energy Storage Systems", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    storage_text = (
        "Energy storage systems are critical for managing the intermittent nature of "
        "renewable energy sources. Lithium-ion batteries currently dominate the market, "
        "but emerging technologies like solid-state batteries and flow batteries offer "
        "promising alternatives. Grid-scale storage capacity reached 45 GW globally in 2024.\n\n"
        "Lithium-ion batteries have seen dramatic cost reductions, falling from over "
        "$1,000 per kWh in 2010 to approximately $139 per kWh in 2023. This cost decline "
        "has been driven by manufacturing scale, improved cell chemistry, and supply chain "
        "optimization.\n\n"
        "Pumped hydroelectric storage remains the largest form of grid-scale energy "
        "storage globally, accounting for approximately 95% of installed storage capacity. "
        "However, its geographic requirements limit deployment options, driving interest "
        "in battery and other technologies.\n\n"
        "Green hydrogen, produced by electrolyzing water using renewable electricity, "
        "serves as a long-duration energy storage medium and can be used in fuel cells, "
        "industrial processes, and transportation. Several countries have announced "
        "national hydrogen strategies, with projected electrolyzer capacity targets "
        "exceeding 100 GW by 2030.\n\n"
        "Compressed air energy storage (CAES) and thermal energy storage represent "
        "additional options for grid-scale applications, each with distinct advantages "
        "in terms of duration, cost, and site requirements."
    )
    pdf.multi_cell(0, 6, storage_text)

    # --- Page 4: Future Outlook ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. Future Outlook and Integration", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    future_text = (
        "The transition to renewable energy systems requires coordinated advances in "
        "generation technology, energy storage, grid infrastructure, and policy frameworks. "
        "The International Energy Agency (IEA) projects that renewable sources will account "
        "for over 50% of global electricity generation by 2030.\n\n"
        "Smart grid technologies are enabling more efficient integration of distributed "
        "renewable resources. These include advanced metering infrastructure (AMI), "
        "distribution automation, and demand response programs that allow consumers to "
        "adjust their energy consumption based on grid conditions.\n\n"
        "The concept of sector coupling links the electricity, heating, and transportation "
        "sectors to maximize the utilization of renewable energy. Electric vehicles serve "
        "as both transportation and potential grid storage through vehicle-to-grid (V2G) "
        "technology.\n\n"
        "Key challenges remain in critical mineral supply chains (lithium, cobalt, rare "
        "earths), grid modernization costs, and workforce development. International "
        "cooperation and sustained policy support are essential for achieving climate "
        "targets outlined in the Paris Agreement.\n\n"
        "Research priorities include next-generation solar cells (perovskites, tandem "
        "cells), advanced wind turbine designs, long-duration storage solutions, and "
        "artificial intelligence applications for grid optimization and energy forecasting."
    )
    pdf.multi_cell(0, 6, future_text)

    # Save
    pdf.output(output_path)
    print(f"✅ Generated sample PDF: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    generate_sample_pdf()
