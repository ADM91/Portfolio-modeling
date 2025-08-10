"""
Portfolio Modeling System Architecture Diagram Generator
Creates a visual representation of the codebase structure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    # Create figure with high DPI for quality
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors for different layers
    colors = {
        'frontend': '#FF6B6B',      # Red-ish
        'api': '#4ECDC4',          # Teal
        'services': '#45B7D1',      # Blue
        'domain': '#96CEB4',        # Green
        'database': '#FECA57',      # Yellow
        'external': '#DDA0DD',      # Plum
        'config': '#F8B500'         # Orange
    }
    
    # Title
    ax.text(8, 11.5, 'Portfolio Modeling System Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Frontend Layer
    frontend_box = FancyBboxPatch((0.5, 9.5), 7, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['frontend'], 
                                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(frontend_box)
    ax.text(4, 10.7, 'Frontend Layer', fontsize=14, fontweight='bold', ha='center')
    ax.text(4, 10.3, 'Streamlit Dashboard', fontsize=11, ha='center')
    ax.text(4, 10.0, '• Interactive portfolio analytics', fontsize=9, ha='center')
    ax.text(4, 9.7, '• Charts, metrics, performance analysis', fontsize=9, ha='center')
    
    # API Client (part of frontend but separate)
    api_client_box = FancyBboxPatch((8.5, 9.5), 3, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['frontend'],
                                    edgecolor='black', linewidth=1, alpha=0.6)
    ax.add_patch(api_client_box)
    ax.text(10, 10.3, 'API Client', fontsize=12, fontweight='bold', ha='center')
    ax.text(10, 9.8, 'Backend Communication', fontsize=9, ha='center')
    
    # FastAPI Layer
    api_box = FancyBboxPatch((2, 7.5), 8, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['api'],
                             edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(api_box)
    ax.text(6, 8.7, 'API Layer (FastAPI)', fontsize=14, fontweight='bold', ha='center')
    ax.text(6, 8.3, 'main.py - Application Entry Point', fontsize=11, ha='center')
    ax.text(4, 7.9, '• metric_router', fontsize=9, ha='center')
    ax.text(4, 7.6, '• data_router', fontsize=9, ha='center')
    ax.text(8, 7.9, '• Portfolio endpoints', fontsize=9, ha='center')
    ax.text(8, 7.6, '• Analytics endpoints', fontsize=9, ha='center')
    
    # Configuration
    config_box = FancyBboxPatch((12, 7.5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['config'],
                                edgecolor='black', linewidth=1, alpha=0.8)
    ax.add_patch(config_box)
    ax.text(13.5, 8.3, 'Configuration', fontsize=12, fontweight='bold', ha='center')
    ax.text(13.5, 7.9, 'config.py', fontsize=10, ha='center')
    ax.text(13.5, 7.6, 'Settings & Assets', fontsize=9, ha='center')
    
    # Services Layer
    services_box = FancyBboxPatch((0.5, 5), 15, 2,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['services'],
                                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(services_box)
    ax.text(8, 6.7, 'Services Layer (Business Logic)', fontsize=14, fontweight='bold', ha='center')
    
    # Individual services
    service_names = [
        ('MetricService', 'Financial calculations\n& analytics'),
        ('DataAcquisitionService', 'External data\nfetching'),
        ('PortfolioService', 'Portfolio\nmanagement'),
        ('TransactionProcessingService', 'Investment action\nprocessing'),
        ('ExcelImportService', 'Data import\nfunctionality'),
        ('TaxLotService', 'Tax calculation\nlogic')
    ]
    
    service_x_positions = [2, 5, 8, 11, 14, 2]
    service_y_positions = [6.2, 6.2, 6.2, 6.2, 6.2, 5.4]
    
    for i, (name, desc) in enumerate(service_names):
        x_pos = service_x_positions[i]
        y_pos = service_y_positions[i] if i < 5 else service_y_positions[i]
        
        service_mini_box = FancyBboxPatch((x_pos-0.8, y_pos-0.3), 1.6, 0.6,
                                          boxstyle="round,pad=0.05",
                                          facecolor='white',
                                          edgecolor='darkblue', linewidth=1, alpha=0.9)
        ax.add_patch(service_mini_box)
        ax.text(x_pos, y_pos+0.1, name, fontsize=8, fontweight='bold', ha='center')
        ax.text(x_pos, y_pos-0.15, desc, fontsize=7, ha='center')
    
    # Domain Layer
    domain_box = FancyBboxPatch((3, 2.5), 10, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['domain'],
                                edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(domain_box)
    ax.text(8, 3.7, 'Domain Layer (Core Business Logic)', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 3.3, '• entities.py', fontsize=10, ha='center')
    ax.text(5, 3.0, '• investment_action.py', fontsize=10, ha='center')
    ax.text(5, 2.7, '• portfolio_holding.py', fontsize=10, ha='center')
    ax.text(11, 3.3, 'Core Entities:', fontsize=10, fontweight='bold', ha='center')
    ax.text(11, 3.0, 'Portfolio, Asset, ActionType', fontsize=9, ha='center')
    ax.text(11, 2.7, 'InvestmentAction, PortfolioHolding', fontsize=9, ha='center')
    
    # Database Layer
    db_box = FancyBboxPatch((5, 0.5), 6, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['database'],
                            edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(db_box)
    ax.text(8, 1.7, 'Database Layer', fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 1.3, 'SQLite Database', fontsize=12, ha='center')
    ax.text(6.5, 0.9, '• Database entities', fontsize=9, ha='center')
    ax.text(6.5, 0.6, '• Data access layer', fontsize=9, ha='center')
    ax.text(9.5, 0.9, '• Portfolio data', fontsize=9, ha='center')
    ax.text(9.5, 0.6, '• Asset prices', fontsize=9, ha='center')
    
    # External Services
    external_box = FancyBboxPatch((12.5, 3), 3, 3,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['external'],
                                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(external_box)
    ax.text(14, 5.5, 'External Services', fontsize=12, fontweight='bold', ha='center')
    ax.text(14, 5.1, 'YFinance Provider', fontsize=10, ha='center')
    ax.text(14, 4.7, '• Stock prices', fontsize=9, ha='center')
    ax.text(14, 4.4, '• Currency rates', fontsize=9, ha='center')
    ax.text(14, 4.1, '• Market data', fontsize=9, ha='center')
    ax.text(14, 3.7, 'Excel Import', fontsize=10, ha='center')
    ax.text(14, 3.4, '• Transaction data', fontsize=9, ha='center')
    ax.text(14, 3.1, '• Portfolio actions', fontsize=9, ha='center')
    
    # Add connection arrows
    arrows = [
        # Frontend to API
        ((4, 9.5), (6, 9.0)),
        # API Client to API
        ((10, 9.5), (8, 9.0)),
        # API to Services
        ((6, 7.5), (8, 7.0)),
        # Services to Domain
        ((8, 5.0), (8, 4.0)),
        # Domain to Database
        ((8, 2.5), (8, 2.0)),
        # External to Services
        ((12.5, 4.5), (11.5, 6.0)),
        # Config to API
        ((12, 8.2), (10, 8.2))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", alpha=0.7, lw=2)
        ax.add_patch(arrow)
    
    # Add data flow labels
    ax.text(1, 8.7, 'User Interaction', fontsize=9, rotation=90, ha='center', style='italic')
    ax.text(14.8, 5.8, 'Data\nSources', fontsize=9, ha='center', style='italic')
    ax.text(8.5, 4.5, 'Business\nLogic', fontsize=9, ha='center', style='italic')
    ax.text(8.5, 1.2, 'Data\nPersistence', fontsize=9, ha='center', style='italic')
    
    # Add technology stack info
    tech_box = FancyBboxPatch((0.5, 0.2), 4, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor='lightgray',
                              edgecolor='black', linewidth=1, alpha=0.8)
    ax.add_patch(tech_box)
    ax.text(2.5, 0.5, 'Tech Stack: Python • FastAPI • Streamlit • SQLite • Pandas • Plotly', 
            fontsize=9, ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def save_diagram():
    """Generate and save the architecture diagram"""
    print("🎨 Generating Portfolio Modeling Architecture Diagram...")
    
    fig = create_architecture_diagram()
    
    # Save as high-quality PNG
    output_file = "portfolio_system_architecture.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Architecture diagram saved as: {output_file}")
    plt.close()
    
    return output_file

if __name__ == "__main__":
    save_diagram()
