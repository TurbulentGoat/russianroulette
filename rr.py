import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import sys

# Suppress any potential warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def get_user_input(simulation_mode):
    print("\nWelcome to the Russian Roulette Simulator!\n")
    
    if simulation_mode == 'probability':
        print("Selected Mode: Law of Total Probability Simulation")
        num_games = get_int("Number of games to simulate (1-100000000): ", 1, 100000000)
        return {
            'num_games': num_games,
            'num_chambers': 6,
            'num_bullets': 1,
            'num_players': 2,
            'continue_after_elimination': False,
            'pulls_per_round': 1,
            'spin_cylinder': False
        }
    else:
        print("Selected Mode: Custom Simulation")
        num_games = get_int("Number of games to simulate (1-100000000): ", 1, 100000000)
        num_chambers = get_int("Number of chambers in the gun (1-100000000): ", 1, 100000000)
        num_bullets = get_int(f"Number of bullets to load (1-{num_chambers}): ", 1, num_chambers)
        num_players = get_int("Number of players to simulate (1-100000000): ", 1, 100000000)
        
        # Option to continue or end the game upon elimination
        continue_after_elimination = get_yes_no("Should the game continue after a player is eliminated? (yes/no): ")
        
        # Number of trigger pulls per player per round
        pulls_per_round = get_int("How many times should each player pull the trigger per round? (1-{num_bullets}): ", 1, num_bullets)
        
        # Option to spin the cylinder between rounds
        spin_cylinder = get_yes_no("Should the cylinder be spun between each round? (yes/no): ")
        
        return {
            'num_games': num_games,
            'num_chambers': num_chambers,
            'num_bullets': num_bullets,
            'num_players': num_players,
            'continue_after_elimination': continue_after_elimination,
            'pulls_per_round': pulls_per_round,
            'spin_cylinder': spin_cylinder
        }

def get_int(prompt, min_val, max_val):
    while True:
        try:
            value = int(input(prompt))
            if value < min_val or value > max_val:
                print(f"Please enter an integer between {min_val} and {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

def select_simulation_mode():
    print("Please select the simulation mode:")
    print("1. Law of Total Probability Simulation")
    print("2. Custom Simulation")
    
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            return 'probability'
        elif choice == '2':
            return 'custom'
        else:
            print("Invalid choice. Please enter 1 or 2.")

def simulate_game(params):
    num_chambers = params['num_chambers']
    num_bullets = params['num_bullets']
    num_players = params['num_players']
    continue_after_elimination = params['continue_after_elimination']
    pulls_per_round = params['pulls_per_round']
    spin_cylinder = params['spin_cylinder']
    
    # Initialize players
    players = {f"Player {i+1}": {'status': 'alive', 'elimination_round': None, 'pulls': 0} for i in range(num_players)}
    active_players = list(players.keys())
    
    # Initialize chambers with bullets
    chambers = [0] * num_chambers
    bullet_positions = np.random.choice(range(num_chambers), size=num_bullets, replace=False)
    for pos in bullet_positions:
        chambers[pos] = 1  # 1 represents a bullet
    
    rounds_without_firing = 0
    consecutive_eliminations = 0
    max_consecutive_eliminations = 0
    
    # Function to get the next chamber index
    def get_next_chamber(current, spin):
        if spin:
            return np.random.randint(0, num_chambers)
        else:
            return (current + 1) % num_chambers
    
    current_chamber = -1  # Start before the first chamber
    
    while active_players and len(bullet_positions) > 0:
        for player in active_players.copy():
            for pull in range(pulls_per_round):
                # Determine next chamber
                current_chamber = get_next_chamber(current_chamber, spin_cylinder)
                
                # Increment player's trigger pulls
                players[player]['pulls'] += 1
                
                if chambers[current_chamber] == 1:
                    # Player is shot
                    players[player]['status'] = 'eliminated'
                    players[player]['elimination_round'] = rounds_without_firing + 1
                    active_players.remove(player)
                    rounds_without_firing += 1
                    consecutive_eliminations += 1
                    if consecutive_eliminations > max_consecutive_eliminations:
                        max_consecutive_eliminations = consecutive_eliminations
                    if not continue_after_elimination:
                        # End the game immediately
                        return {
                            'rounds_without_firing': rounds_without_firing,
                            'max_consecutive_eliminations': max_consecutive_eliminations,
                            'player_stats': players,
                            'bullet_positions': bullet_positions
                        }
                    break  # Move to next player after elimination
                else:
                    rounds_without_firing += 1
            if not active_players:
                break  # No active players left
    
        # Reset consecutive eliminations if no elimination in this round
        if consecutive_eliminations > 0:
            consecutive_eliminations = 0
    
    return {
        'rounds_without_firing': rounds_without_firing,
        'max_consecutive_eliminations': max_consecutive_eliminations,
        'player_stats': players,
        'bullet_positions': bullet_positions
    }

def run_simulations(params):
    num_games = params['num_games']
    simulation_results = []
    
    for game in range(num_games):
        result = simulate_game(params)
        simulation_results.append(result)
    
    return simulation_results

def analyze_results(simulation_results, params):
    num_games = params['num_games']
    num_players = params['num_players']
    num_chambers = params['num_chambers']
    
    # Initialize data structures
    total_rounds_without_firing = 0
    max_consecutive_eliminations_list = []
    player_survival = {f"Player {i+1}": [] for i in range(num_players)}
    bullet_heatmap = np.zeros(params['num_chambers'])
    
    for result in simulation_results:
        total_rounds_without_firing += result['rounds_without_firing']
        max_consecutive_eliminations_list.append(result['max_consecutive_eliminations'])
        for player, stats in result['player_stats'].items():
            elimination_round = stats['elimination_round']
            if elimination_round is not None:
                player_survival[player].append(elimination_round)
            else:
                player_survival[player].append(params['num_chambers'])  # Survived all rounds
        # Update heatmap
        for pos in result['bullet_positions']:
            bullet_heatmap[pos] += 1
    
    # Calculate averages
    avg_rounds_without_firing = total_rounds_without_firing / num_games
    avg_max_consecutive_eliminations = np.mean(max_consecutive_eliminations_list)
    
    # Player survival analysis
    player_average_survival = {player: np.mean(rounds) for player, rounds in player_survival.items()}
    
    # Heatmap data normalization
    bullet_heatmap_percentage = (bullet_heatmap / num_games) * 100
    
    analysis = {
        'avg_rounds_without_firing': avg_rounds_without_firing,
        'avg_max_consecutive_eliminations': avg_max_consecutive_eliminations,
        'player_average_survival': player_average_survival,
        'bullet_heatmap_percentage': bullet_heatmap_percentage
    }
    
    return analysis

def generate_plots(analysis, params, simulation_results):
    num_games = params['num_games']
    num_players = params['num_players']
    num_chambers = params['num_chambers']
    
    # Create a directory to save plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Histogram of Rounds Without Firing
    plt.figure(figsize=(10,6))
    sns.histplot([result['rounds_without_firing'] for result in simulation_results], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Rounds Without Firing', fontsize=16)
    plt.xlabel('Rounds Without Firing', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/rounds_without_firing_hist.png')
    plt.close()
    
    # 2. Bar Chart of Player Survival Times (Enhanced)
    players = list(analysis['player_average_survival'].keys())
    survival_times = list(analysis['player_average_survival'].values())
    
    plt.figure(figsize=(12,8))
    sns.barplot(x=players, y=survival_times, palette='viridis')
    plt.title('Average Survival Rounds per Player', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Average Survival Rounds', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/player_survival_bar.png')
    plt.close()
    
    # 3. Bar Chart for Bullet Position Distribution (%)
    bullet_heatmap = analysis['bullet_heatmap_percentage']
    chambers = range(1, num_chambers + 1)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(chambers), y=bullet_heatmap, palette='magma')
    plt.title('Bullet Position Distribution (%)', fontsize=16)
    plt.xlabel('Chamber Number', fontsize=14)
    plt.ylabel('Bullet Frequency (%)', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/bullet_position_distribution_bar.png')
    plt.close()
    
    # 4. Boxplot of Max Consecutive Eliminations
    max_consec_elims = [result['max_consecutive_eliminations'] for result in simulation_results]
    plt.figure(figsize=(8,6))
    sns.boxplot(y=max_consec_elims, color='lightgreen')
    plt.title('Boxplot of Max Consecutive Eliminations per Game', fontsize=16)
    plt.ylabel('Max Consecutive Eliminations', fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/max_consecutive_elims_box.png')
    plt.close()
    
    # 5. Pie Chart of Game Outcomes (Enhanced)
    # For simplicity, assuming two outcomes: early ended vs continued
    early_end = sum(1 for res in simulation_results if res['max_consecutive_eliminations'] > 0)
    continued = num_games - early_end
    labels = ['Early Ended Games', 'Continued Games']
    sizes = [early_end, continued]
    colors = ['gold', 'lightcoral']
    explode = (0.05, 0)  # Slightly explode the first slice
    
    plt.figure(figsize=(8,8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, shadow=True, startangle=140)
    plt.title('Game Outcomes', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig('plots/game_outcomes_pie.png')
    plt.close()

def generate_pdf_report(analysis, params, simulation_results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 10, "Russian Roulette Simulation Report", ln=True, align='C')
    pdf.ln(10)
    
    # Introduction
    pdf.set_font("Arial", size=12)
    intro_text = f"""
    This report presents the results of {params['num_games']} simulations of Russian Roulette with the following parameters:
    
    - **Number of Chambers:** {params['num_chambers']}
    - **Number of Bullets Loaded:** {params['num_bullets']}
    - **Number of Players:** {params['num_players']}
    - **Continue After Elimination:** {"Yes" if params['continue_after_elimination'] else "No"}
    - **Trigger Pulls Per Player Per Round:** {params['pulls_per_round']}
    - **Cylinder Spun Between Rounds:** {"Yes" if params['spin_cylinder'] else "No"}
    """
    pdf.multi_cell(0, 10, intro_text)
    
    # Add plots
    plot_files = [
        ('rounds_without_firing_hist.png', 'Distribution of Rounds Without Firing'),
        ('player_survival_bar.png', 'Average Survival Rounds per Player'),
        ('bullet_position_distribution_bar.png', 'Bullet Position Distribution (%)'),
        ('max_consecutive_elims_box.png', 'Max Consecutive Eliminations per Game'),
        ('game_outcomes_pie.png', 'Game Outcomes')
    ]
    
    for file, title in plot_files:
        if os.path.exists(f"plots/{file}"):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, title, ln=True, align='C')
            pdf.ln(10)
            # Calculate the image dimensions to fit the page
            # FPDF uses (x, y, w, h) for image placement
            # We'll set width to 190 mm and let height adjust automatically
            pdf.image(f"plots/{file}", x=10, y=None, w=190)
        else:
            print(f"Warning: Plot file {file} not found.")
    
    # Add Statistics Summary
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, "Statistics Summary", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    
    summary_text = f"""
    - **Average Rounds Without Firing:** {analysis['avg_rounds_without_firing']:.2f}
    - **Average Max Consecutive Eliminations:** {analysis['avg_max_consecutive_eliminations']:.2f}
    """
    pdf.multi_cell(0, 10, summary_text)
    
    # Bullet Position Distribution as a Table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Bullet Position Distribution (%)", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    
    # Create a DataFrame for bullet positions
    chambers = range(1, params['num_chambers'] + 1)
    bullet_data = {
        'Chamber': list(chambers),
        'Bullet Frequency (%)': [f"{freq:.2f}" for freq in analysis['bullet_heatmap_percentage']]
    }
    bullet_df = pd.DataFrame(bullet_data)
    
    # Define column widths
    col_widths = [40, 40]  # Adjust as needed
    
    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(col_widths[0], 10, "Chamber", border=1, align='C')
    pdf.cell(col_widths[1], 10, "Bullet Frequency (%)", border=1, align='C')
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Arial", size=12)
    for index, row in bullet_df.iterrows():
        pdf.cell(col_widths[0], 10, f"Chamber {row['Chamber']}", border=1, align='C')
        pdf.cell(col_widths[1], 10, f"{row['Bullet Frequency (%)']}", border=1, align='C')
        pdf.ln()
    
    # Player Survival Times
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Average Survival Rounds per Player", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    
    # Define column widths
    player_col_width = 60
    survival_col_width = 40
    
    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(player_col_width, 10, "Player", border=1, align='C')
    pdf.cell(survival_col_width, 10, "Average Survival Rounds", border=1, align='C')
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Arial", size=12)
    for player, avg_survival in analysis['player_average_survival'].items():
        pdf.cell(player_col_width, 10, f"{player}", border=1, align='C')
        pdf.cell(survival_col_width, 10, f"{avg_survival:.2f}", border=1, align='C')
        pdf.ln()
    
    # Add additional statistics if needed
    
    # Save PDF
    if not os.path.exists('reports'):
        os.makedirs('reports')
    pdf.output("reports/Russian_Roulette_Simulation_Report.pdf")
    print("\nPDF report generated at 'reports/Russian_Roulette_Simulation_Report.pdf'.")

def main():
    # Step 1: Select Simulation Mode
    simulation_mode = select_simulation_mode()
    
    # Step 2: Get User Inputs Based on Mode
    params = get_user_input(simulation_mode)
    
    # Step 3: Run Simulations
    print("\nRunning simulations...")
    simulation_results = run_simulations(params)
    print("Simulations completed.")
    
    # Step 4: Analyze Results
    print("Analyzing results...")
    analysis = analyze_results(simulation_results, params)
    print("Analysis completed.")
    
    # Step 5: Generate Plots
    print("Generating plots...")
    generate_plots(analysis, params, simulation_results)
    print("Plots generated.")
    
    # Step 6: Generate PDF Report
    print("Generating PDF report...")
    generate_pdf_report(analysis, params, simulation_results)
    print("Report generation completed.")

if __name__ == "__main__":
    main()
