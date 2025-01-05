import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import sys
import datetime
import multiprocessing
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

# For Real-Time Interactive Mode (GUI)
import tkinter as tk
from tkinter import messagebox

# For Bayesian Updates
from scipy.stats import beta

def get_user_input(simulation_mode):
    print("\nWelcome to the Enhanced Russian Roulette Simulator!\n")
    
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
            'spin_cylinder': False,
            'betting': False,
            'interactive': False
        }
    elif simulation_mode == 'betting':
        print("Selected Mode: Betting and Wagering System")
        num_games = get_int("Number of games to simulate (1-100000000): ", 1, 100000000)
        num_chambers = get_int("Number of chambers in the gun (1-100): ", 1, 100)  # Reduced for practicality
        num_bullets = get_int(f"Number of bullets to load (1-{num_chambers}): ", 1, num_chambers)
        num_players = get_int("Number of players to simulate (1-100): ", 1, 100)  # Reduced for practicality
        
        # Betting parameters
        starting_money = get_float("Starting money per player (e.g., 100): ", 1, 1000000)
        bet_amount = get_float("Bet amount per game (e.g., 10): ", 1, starting_money)
        
        # Option to continue or end the game upon elimination
        continue_after_elimination = get_yes_no("Should the game continue after a player is eliminated? (yes/no): ")
        
        # Number of trigger pulls per player per round
        pulls_per_round = get_int(f"How many times should each player pull the trigger per round? (1-{num_bullets}): ", 1, num_bullets)
        
        # Option to spin the cylinder between rounds
        spin_cylinder = get_yes_no("Should the cylinder be spun between each round? (yes/no): ")
        
        return {
            'num_games': num_games,
            'num_chambers': num_chambers,
            'num_bullets': num_bullets,
            'num_players': num_players,
            'continue_after_elimination': continue_after_elimination,
            'pulls_per_round': pulls_per_round,
            'spin_cylinder': spin_cylinder,
            'betting': True,
            'starting_money': starting_money,
            'bet_amount': bet_amount,
            'interactive': False
        }
    elif simulation_mode == 'real_time':
        print("Selected Mode: Real-Time Interactive Game")
        # Parameters for interactive mode can be fixed or configurable
        num_chambers = get_int("Number of chambers in the gun (1-10): ", 1, 10)
        num_bullets = get_int(f"Number of bullets to load (1-{num_chambers}): ", 1, num_chambers)
        num_players = get_int("Number of players (1-4): ", 1, 4)
        
        # Option to spin the cylinder between rounds
        spin_cylinder = get_yes_no("Should the cylinder be spun between each round? (yes/no): ")
        
        return {
            'num_games': 1,  # Not applicable for interactive mode
            'num_chambers': num_chambers,
            'num_bullets': num_bullets,
            'num_players': num_players,
            'continue_after_elimination': False,  # Not applicable
            'pulls_per_round': 1,  # Fixed for interactive mode
            'spin_cylinder': spin_cylinder,
            'betting': False,
            'interactive': True
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
        pulls_per_round = get_int(f"How many times should each player pull the trigger per round? (1-{num_bullets}): ", 1, num_bullets)
        
        # Option to spin the cylinder between rounds
        spin_cylinder = get_yes_no("Should the cylinder be spun between each round? (yes/no): ")
        
        return {
            'num_games': num_games,
            'num_chambers': num_chambers,
            'num_bullets': num_bullets,
            'num_players': num_players,
            'continue_after_elimination': continue_after_elimination,
            'pulls_per_round': pulls_per_round,
            'spin_cylinder': spin_cylinder,
            'betting': False,
            'interactive': False
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

def get_float(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if value < min_val or value > max_val:
                print(f"Please enter a number between {min_val} and {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

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
    print("2. Betting and Wagering System")
    print("3. Real-Time Interactive Game")
    print("4. Custom Simulation")
    
    while True:
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice == '1':
            return 'probability'
        elif choice == '2':
            return 'betting'
        elif choice == '3':
            return 'real_time'
        elif choice == '4':
            return 'custom'
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def simulate_game(params):
    num_chambers = params['num_chambers']
    num_bullets = params['num_bullets']
    num_players = params['num_players']
    continue_after_elimination = params['continue_after_elimination']
    pulls_per_round = params['pulls_per_round']
    spin_cylinder = params['spin_cylinder']
    betting = params['betting']
    interactive = params['interactive']
    
    # Initialize players
    players = {
        f"Player {i+1}": {
            'status': 'alive',
            'elimination_round': None,
            'pulls': 0,
            'money': params.get('starting_money', 0)
        } for i in range(num_players)
    }
    active_players = list(players.keys())
    
    # Initialize chambers with bullets
    chambers = [0] * num_chambers
    bullet_positions = np.random.choice(range(num_chambers), size=num_bullets, replace=False)
    for pos in bullet_positions:
        chambers[pos] = 1  # 1 represents a bullet
    
    rounds_without_firing = 0
    consecutive_eliminations = 0
    max_consecutive_eliminations = 0
    time_to_first_elimination = None
    
    # Function to get the next chamber index
    def get_next_chamber(current, spin):
        if spin:
            return np.random.randint(0, num_chambers)
        else:
            return (current + 1) % num_chambers
    
    current_chamber = -1  # Start before the first chamber
    
    while active_players and sum(chambers) > 0:
        for player in active_players.copy():
            for _ in range(pulls_per_round):
                # Determine next chamber
                current_chamber = get_next_chamber(current_chamber, spin_cylinder)
                
                # Increment player's trigger pulls
                players[player]['pulls'] += 1
                
                # Check for bullet
                if chambers[current_chamber] == 1:
                    # Player is shot
                    players[player]['status'] = 'eliminated'
                    players[player]['elimination_round'] = rounds_without_firing + 1
                    active_players.remove(player)
                    rounds_without_firing += 1
                    consecutive_eliminations += 1
                    max_consecutive_eliminations = max(consecutive_eliminations, max_consecutive_eliminations)
                    
                    if time_to_first_elimination is None:
                        time_to_first_elimination = rounds_without_firing
                    
                    if betting:
                        # Handle betting: player loses bet amount
                        players[player]['money'] -= params['bet_amount']
                        # Distribute bet to other players
                        for other in active_players:
                            players[other]['money'] += params['bet_amount'] / len(active_players)
                    
                    if not continue_after_elimination:
                        # End the game immediately
                        return {
                            'rounds_without_firing': rounds_without_firing,
                            'max_consecutive_eliminations': max_consecutive_eliminations,
                            'player_stats': players,
                            'bullet_positions': bullet_positions,
                            'time_to_first_elimination': time_to_first_elimination,
                            'betting': betting
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
        'bullet_positions': bullet_positions,
        'time_to_first_elimination': time_to_first_elimination,
        'betting': betting
    }

def run_simulations(params):
    num_games = params['num_games']
    simulation_results = []
    
    # Utilize multiprocessing for performance enhancement
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)
    
    try:
        # Prepare parameters for each simulation
        simulations = [params for _ in range(num_games)]
        simulation_results = pool.map(simulate_game, simulations)
    finally:
        pool.close()
        pool.join()
    
    return simulation_results

def analyze_results(simulation_results, params):
    num_games = params['num_games']
    num_players = params['num_players']
    num_chambers = params['num_chambers']
    
    # Initialize data structures
    rounds_without_firing_list = []
    max_consecutive_eliminations_list = []
    player_survival = {f"Player {i+1}": [] for i in range(num_players)}
    bullet_heatmap = np.zeros(params['num_chambers'])
    games_all_survived = 0
    games_all_eliminated = 0
    time_to_first_elimination_list = []
    
    # For Bayesian Updates (simple example using Beta distribution)
    # Assuming prior alpha=1, beta=1 for each player
    bayesian_prior = {
        f"Player {i+1}": {'alpha': 1, 'beta': 1} for i in range(num_players)
    }
    
    for result in simulation_results:
        rounds_without_firing_list.append(result['rounds_without_firing'])
        max_consecutive_eliminations_list.append(result['max_consecutive_eliminations'])
        
        first_elim = result['time_to_first_elimination']
        if first_elim is not None:
            time_to_first_elimination_list.append(first_elim)
        
        eliminations = 0
        for player, stats in result['player_stats'].items():
            elimination_round = stats['elimination_round']
            if elimination_round is not None:
                player_survival[player].append(elimination_round)
                eliminations += 1
                # Bayesian Update: Treat elimination as a 'failure'
                bayesian_prior[player]['beta'] += 1
            else:
                player_survival[player].append(params['num_chambers'])  # Survived all rounds
                # Bayesian Update: Treat survival as a 'success'
                bayesian_prior[player]['alpha'] += 1
        
        if eliminations == 0:
            games_all_survived += 1
        if eliminations == num_players:
            games_all_eliminated += 1
        
        # Update heatmap
        for pos in result['bullet_positions']:
            bullet_heatmap[pos] += 1
    
    # Calculate averages and additional statistics
    avg_rounds_without_firing = np.mean(rounds_without_firing_list)
    median_rounds_without_firing = np.median(rounds_without_firing_list)
    std_rounds_without_firing = np.std(rounds_without_firing_list)
    percentile25_rounds = np.percentile(rounds_without_firing_list, 25)
    percentile75_rounds = np.percentile(rounds_without_firing_list, 75)
    
    avg_max_consecutive_eliminations = np.mean(max_consecutive_eliminations_list)
    median_max_consecutive_eliminations = np.median(max_consecutive_eliminations_list)
    std_max_consecutive_eliminations = np.std(max_consecutive_eliminations_list)
    percentile25_max_consec = np.percentile(max_consecutive_eliminations_list, 25)
    percentile75_max_consec = np.percentile(max_consecutive_eliminations_list, 75)
    
    # Player survival analysis
    player_average_survival = {
        player: np.mean(rounds) for player, rounds in player_survival.items()
    }
    player_median_survival = {
        player: np.median(rounds) for player, rounds in player_survival.items()
    }
    player_std_survival = {
        player: np.std(rounds) for player, rounds in player_survival.items()
    }
    player_survival_rate = {
        player: (
            sum(1 for round_survived in rounds if round_survived >= params['num_chambers']) / num_games
        ) * 100 for player, rounds in player_survival.items()
    }
    
    # Heatmap data normalization
    bullet_heatmap_percentage = (bullet_heatmap / num_games) * 100
    
    # Overall game statistics
    games_all_survived_percentage = (games_all_survived / num_games) * 100
    games_all_eliminated_percentage = (games_all_eliminated / num_games) * 100
    avg_time_to_first_elimination = (
        np.mean(time_to_first_elimination_list) if time_to_first_elimination_list else None
    )
    
    # Bayesian Estimates
    bayesian_estimates = {
        player: beta.mean(a=bayesian_prior[player]['alpha'], b=bayesian_prior[player]['beta'])
        for player in bayesian_prior
    }
    
    analysis = {
        'avg_rounds_without_firing': avg_rounds_without_firing,
        'median_rounds_without_firing': median_rounds_without_firing,
        'std_rounds_without_firing': std_rounds_without_firing,
        'percentile25_rounds': percentile25_rounds,
        'percentile75_rounds': percentile75_rounds,
        
        'avg_max_consecutive_eliminations': avg_max_consecutive_eliminations,
        'median_max_consecutive_eliminations': median_max_consecutive_eliminations,
        'std_max_consecutive_eliminations': std_max_consecutive_eliminations,
        'percentile25_max_consec': percentile25_max_consec,
        'percentile75_max_consec': percentile75_max_consec,
        
        'player_average_survival': player_average_survival,
        'player_median_survival': player_median_survival,
        'player_std_survival': player_std_survival,
        'player_survival_rate': player_survival_rate,
        
        'bullet_heatmap_percentage': bullet_heatmap_percentage,
        
        'games_all_survived': games_all_survived,
        'games_all_eliminated': games_all_eliminated,
        'games_all_survived_percentage': games_all_survived_percentage,
        'games_all_eliminated_percentage': games_all_eliminated_percentage,
        'avg_time_to_first_elimination': avg_time_to_first_elimination,
        
        'bayesian_estimates': bayesian_estimates
    }
    
    return analysis

def generate_plots(analysis, params, simulation_results):
    num_games = params['num_games']
    num_players = params['num_players']
    num_chambers = params['num_chambers']
    num_bullets = params['num_bullets']  # Extract num_bullets from params
    
    # Create a directory to save plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Histogram of Rounds Without Firing
    plt.figure(figsize=(10,6))
    sns.histplot(
        [result['rounds_without_firing'] for result in simulation_results],
        bins=20,
        kde=True,
        color='skyblue'
    )
    plt.title('Distribution of Rounds Without Firing', fontsize=16)
    plt.xlabel('Rounds Without Firing', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/rounds_without_firing_hist.png')
    plt.close()
    
    # 2. Boxplot of Rounds Without Firing
    plt.figure(figsize=(8,6))
    sns.boxplot(
        y=[result['rounds_without_firing'] for result in simulation_results],
        color='lightblue'
    )
    plt.title('Boxplot of Rounds Without Firing', fontsize=16)
    plt.ylabel('Rounds Without Firing', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/rounds_without_firing_box.png')
    plt.close()
    
    # 3. Bar Chart of Player Survival Times (Average)
    players = list(analysis['player_average_survival'].keys())
    survival_times = list(analysis['player_average_survival'].values())
    
    plt.figure(figsize=(12,8))
    sns.barplot(
        x=players,
        y=survival_times,
        palette='viridis'
    )
    plt.title('Average Survival Rounds per Player', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Average Survival Rounds', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/player_survival_avg_bar.png')
    plt.close()
    
    # 4. Boxplot of Player Survival Times
    plt.figure(figsize=(12,8))
    player_survival_data = [analysis['player_average_survival'][player] for player in players]
    sns.boxplot(
        data=player_survival_data,
        palette='pastel'
    )
    plt.title('Boxplot of Player Survival Rounds', fontsize=16)
    plt.ylabel('Survival Rounds', fontsize=14)
    plt.xticks(ticks=[], labels=[])  # Hide x-axis labels
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/player_survival_box.png')
    plt.close()
    
    # 5. Bar Chart for Bullet Position Distribution (%)
    bullet_heatmap = analysis['bullet_heatmap_percentage']
    chambers = range(1, num_chambers + 1)
    
    plt.figure(figsize=(12,6))
    sns.barplot(
        x=list(chambers),
        y=bullet_heatmap,
        palette='magma'
    )
    plt.title('Bullet Position Distribution (%)', fontsize=16)
    plt.xlabel('Chamber Number', fontsize=14)
    plt.ylabel('Bullet Frequency (%)', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/bullet_position_distribution_bar.png')
    plt.close()
    
    # 6. Boxplot of Max Consecutive Eliminations
    max_consec_elims = [result['max_consecutive_eliminations'] for result in simulation_results]
    plt.figure(figsize=(8,6))
    sns.boxplot(
        y=max_consec_elims,
        color='lightgreen'
    )
    plt.title('Boxplot of Max Consecutive Eliminations per Game', fontsize=16)
    plt.ylabel('Max Consecutive Eliminations', fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/max_consecutive_elims_box.png')
    plt.close()
    
    # 7. Histogram of Time to First Elimination
    time_to_first_elimination = [
        result['time_to_first_elimination'] for result in simulation_results
        if result['time_to_first_elimination'] is not None
    ]
    if time_to_first_elimination:
        plt.figure(figsize=(10,6))
        sns.histplot(
            time_to_first_elimination,
            bins=20,
            kde=True,
            color='salmon'
        )
        plt.title('Distribution of Time to First Elimination', fontsize=16)
        plt.xlabel('Rounds Until First Elimination', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('plots/time_to_first_elimination_hist.png')
        plt.close()
    
    # 8. Survival Rate Bar Chart
    survival_rates = list(analysis['player_survival_rate'].values())
    plt.figure(figsize=(12,8))
    sns.barplot(
        x=players,
        y=survival_rates,
        palette='coolwarm'
    )
    plt.title('Survival Rate per Player (%)', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Survival Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/player_survival_rate_bar.png')
    plt.close()
    
    # 9. Bayesian Estimates Bar Chart
    bayesian_estimates = analysis['bayesian_estimates']
    plt.figure(figsize=(12,8))
    sns.barplot(
        x=list(bayesian_estimates.keys()),
        y=list(bayesian_estimates.values()),
        palette='viridis'
    )
    plt.title('Bayesian Estimated Survival Probabilities', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Estimated Survival Probability', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)  # Beta distribution mean between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/bayesian_estimates_bar.png')
    plt.close()
    
    # 10. Probability Distribution Comparison
    # Theoretical probability of survival with fixed spins
    theoretical_prob = (num_chambers - num_bullets) / num_chambers
    theoretical_mean_rounds = 1 / theoretical_prob  # Correct theoretical mean for geometric distribution
    plt.figure(figsize=(10,6))
    sns.histplot(
        [result['rounds_without_firing'] for result in simulation_results],
        bins=20,
        kde=True,
        color='skyblue',
        label='Simulation'
    )
    plt.axvline(
        x=theoretical_mean_rounds,
        color='red',
        linestyle='--',
        label='Theoretical Mean'
    )
    plt.title('Comparison of Simulation and Theoretical Distribution', fontsize=16)
    plt.xlabel('Rounds Without Firing', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/probability_distribution_comparison.png')
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
    if params['interactive']:
        intro_text = f"""
        This interactive Russian Roulette game was played with the following settings:
        
        - **Number of Chambers:** {params['num_chambers']}
        - **Number of Bullets Loaded:** {params['num_bullets']}
        - **Number of Players:** {params['num_players']}
        - **Cylinder Spun Between Rounds:** {"Yes" if params['spin_cylinder'] else "No"}
        """
    elif params['betting']:
        intro_text = f"""
        This report presents the results of {params['num_games']} simulations of Russian Roulette with Betting and Wagering. The simulation was conducted with the following parameters:
        
        - **Number of Chambers:** {params['num_chambers']}
        - **Number of Bullets Loaded:** {params['num_bullets']}
        - **Number of Players:** {params['num_players']}
        - **Continue After Elimination:** {"Yes" if params['continue_after_elimination'] else "No"}
        - **Trigger Pulls Per Player Per Round:** {params['pulls_per_round']}
        - **Cylinder Spun Between Rounds:** {"Yes" if params['spin_cylinder'] else "No"}
        - **Starting Money per Player:** {params['starting_money']}
        - **Bet Amount per Game:** {params['bet_amount']}
        """
    else:
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
        ('rounds_without_firing_box.png', 'Boxplot of Rounds Without Firing'),
        ('player_survival_avg_bar.png', 'Average Survival Rounds per Player'),
        ('player_survival_box.png', 'Boxplot of Player Survival Rounds'),
        ('player_survival_rate_bar.png', 'Survival Rate per Player (%)'),
        ('bayesian_estimates_bar.png', 'Bayesian Estimated Survival Probabilities'),
        ('bullet_position_distribution_bar.png', 'Bullet Position Distribution (%)'),
        ('max_consecutive_elims_box.png', 'Boxplot of Max Consecutive Eliminations per Game'),
        ('time_to_first_elimination_hist.png', 'Distribution of Time to First Elimination'),
        ('probability_distribution_comparison.png', 'Comparison of Simulation and Theoretical Distribution')
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
    - **Median Rounds Without Firing:** {analysis['median_rounds_without_firing']}
    - **Standard Deviation:** {analysis['std_rounds_without_firing']:.2f}
    - **25th Percentile:** {analysis['percentile25_rounds']}
    - **75th Percentile:** {analysis['percentile75_rounds']}
    
    - **Average Max Consecutive Eliminations:** {analysis['avg_max_consecutive_eliminations']:.2f}
    - **Median Max Consecutive Eliminations:** {analysis['median_max_consecutive_eliminations']}
    - **Standard Deviation:** {analysis['std_max_consecutive_eliminations']:.2f}
    - **25th Percentile:** {analysis['percentile25_max_consec']}
    - **75th Percentile:** {analysis['percentile75_max_consec']}
    
    - **Games with All Players Survived:** {analysis['games_all_survived']} ({analysis['games_all_survived_percentage']:.2f}%)
    - **Games with All Players Eliminated:** {analysis['games_all_eliminated']} ({analysis['games_all_eliminated_percentage']:.2f}%)
    - **Average Time to First Elimination:** {"N/A" if analysis['avg_time_to_first_elimination'] is None else f"{analysis['avg_time_to_first_elimination']:.2f} rounds"}
    
    - **Bayesian Estimated Survival Probabilities:**
    """
    for player, estimate in analysis['bayesian_estimates'].items():
        summary_text += f"\n  - {player}: {estimate * 100:.2f}%"
    
    pdf.multi_cell(0, 10, summary_text)
    
    # Bullet Position Distribution as a Table
    pdf.set_font("Arial", 'B', 14)
    pdf.ln(10)  # Adds vertical space before the title
    pdf.cell(0, 10, "Bullet Position Distribution (%)", ln=True, align='L')
    pdf.ln(5)  # Adds a small space after the title
    pdf.set_font("Arial", size=12)
    
    # Create a DataFrame for bullet positions
    chambers = range(1, params['num_chambers'] + 1)
    bullet_data = {
        'Chamber': [f"Chamber {chamber}" for chamber in chambers],
        'Bullet Frequency (%)': [f"{freq:.2f}%" for freq in analysis['bullet_heatmap_percentage']]
    }
    bullet_df = pd.DataFrame(bullet_data)
    
    # Define column widths based on content length
    col_widths = [50, 50]  # Increased widths for better content fitting
    
    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(col_widths[0], 10, "Chamber", border=1, align='C')
    pdf.cell(col_widths[1], 10, "Bullet Frequency (%)", border=1, align='C')
    pdf.ln()  # Move to the next line after the header
    
    # Table Rows
    pdf.set_font("Arial", size=12)
    for index, row in bullet_df.iterrows():
        pdf.cell(col_widths[0], 10, row['Chamber'], border=1, align='C')
        pdf.cell(col_widths[1], 10, row['Bullet Frequency (%)'], border=1, align='C')
        pdf.ln()  # Move to the next line after each row
    
    # Player Survival Times
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Player Survival Statistics", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    
    # Player Survival Table
    players = list(analysis['player_average_survival'].keys())
    player_data = {
        'Player': players,
        'Average Survival Rounds': [f"{analysis['player_average_survival'][player]:.2f}" for player in players],
        'Median Survival Rounds': [f"{analysis['player_median_survival'][player]}" for player in players],
        'Std Dev': [f"{analysis['player_std_survival'][player]:.2f}" for player in players],
        'Survival Rate (%)': [f"{analysis['player_survival_rate'][player]:.2f}%" for player in players]
    }
    player_df = pd.DataFrame(player_data)
    
    # Define column widths
    player_col_width = 40
    avg_col_width = 40
    median_col_width = 35
    std_col_width = 35
    rate_col_width = 35
    
    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(player_col_width, 15, "Player", border=1, align='C')
    pdf.cell(avg_col_width, 15, "Average Survival", border=1, align='C')
    pdf.cell(median_col_width, 15, "Median Survival", border=1, align='C')
    pdf.cell(std_col_width, 15, "Std Dev", border=1, align='C')
    pdf.cell(rate_col_width, 15, "Survival Rate (%)", border=1, align='C')
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Arial", size=12)
    for index, row in player_df.iterrows():
        pdf.cell(player_col_width, 10, row['Player'], border=1, align='C')
        pdf.cell(avg_col_width, 10, row['Average Survival Rounds'], border=1, align='C')
        pdf.cell(median_col_width, 10, row['Median Survival Rounds'], border=1, align='C')
        pdf.cell(std_col_width, 10, row['Std Dev'], border=1, align='C')
        pdf.cell(rate_col_width, 10, row['Survival Rate (%)'], border=1, align='C')
        pdf.ln()
    
    # Betting Results (if applicable)
    if params['betting']:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Betting and Wagering Results", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        
        betting_summary = ""
        for player, stats in simulation_results[0]['player_stats'].items():
            if params['num_games'] > 0:
                final_money = stats.get('money', 0)
                betting_summary += f"- **{player}:** Final Money = {final_money:.2f}\n"
        
        pdf.multi_cell(0, 10, betting_summary)
    
    # Save PDF
    if not os.path.exists('reports'):
        os.makedirs('reports')

    timestamp = datetime.datetime.now().strftime('%d.%m_%I%M%p')
    pdf_filename = f"reports/Report_{timestamp}_{params['num_games']}.pdf"
    pdf.output(pdf_filename)
    print(f"\nPDF report generated at '{pdf_filename}'.")

def interactive_game(params):
    # Simple GUI using tkinter for real-time interactive game
    num_chambers = params['num_chambers']
    num_bullets = params['num_bullets']
    num_players = params['num_players']
    spin_cylinder = params['spin_cylinder']
    
    # Initialize chambers
    chambers = [0] * num_chambers
    bullet_positions = np.random.choice(range(num_chambers), size=num_bullets, replace=False)
    for pos in bullet_positions:
        chambers[pos] = 1  # 1 represents a bullet
    
    current_chamber = -1  # Start before the first chamber
    players = {f"Player {i+1}": {'status': 'alive'} for i in range(num_players)}
    active_players = list(players.keys())
    player_turn = 0  # Index of the current player
    
    # Function to get the next chamber based on spinning
    def get_next_chamber(current, spin):
        if spin:
            return np.random.randint(0, num_chambers)
        else:
            return (current + 1) % num_chambers
    
    # Function to handle shooting
    def shoot():
        nonlocal current_chamber, player_turn, active_players
        if not active_players:
            messagebox.showinfo("Game Over", "All players have been eliminated.")
            root.destroy()
            return
        
        current_player = active_players[player_turn]
        current_chamber = get_next_chamber(current_chamber, spin_cylinder)
        
        if chambers[current_chamber] == 1:
            # Player is shot
            players[current_player]['status'] = 'eliminated'
            messagebox.showwarning("Elimination", f"{current_player} has been eliminated!")
            active_players.remove(current_player)
            if not active_players:
                messagebox.showinfo("Game Over", "All players have been eliminated.")
                root.destroy()
                return
            else:
                # Adjust player_turn
                player_turn = player_turn % len(active_players)
        else:
            messagebox.showinfo("Safe", f"{current_player} survived this round!")
            # Move to next player
            player_turn = (player_turn + 1) % len(active_players)
    
    # Function to handle spinning via mouse wheel
    def spin(event):
        nonlocal spin_cylinder
        spin_cylinder = not spin_cylinder
        status = "Spun" if spin_cylinder else "Sequential"
        messagebox.showinfo("Cylinder Spun", f"Cylinder spinning is now set to: {status}")
    
    # Initialize tkinter window
    root = tk.Tk()
    root.title("Russian Roulette - Interactive Mode")
    
    # Instructions
    instructions = tk.Label(
        root,
        text="Click 'Shoot' to pull the trigger.\nUse the mouse wheel to toggle cylinder spinning.",
        font=("Arial", 14)
    )
    instructions.pack(pady=10)
    
    # Shoot Button
    shoot_button = tk.Button(
        root,
        text="Shoot",
        font=("Arial", 16),
        command=shoot,
        bg='red',
        fg='white'
    )
    shoot_button.pack(pady=20)
    
    # Bind mouse wheel to spin
    root.bind("<MouseWheel>", spin)  # Windows
    root.bind("<Button-4>", spin)    # Linux scroll up
    root.bind("<Button-5>", spin)    # Linux scroll down
    
    # Player Status Display
    status_frame = tk.Frame(root)
    status_frame.pack(pady=10)
    
    for player in players:
        status_label = tk.Label(
            status_frame,
            text=f"{player}: Alive",
            font=("Arial", 12)
        )
        status_label.pack(anchor='w')
        players[player]['label'] = status_label
    
    # Update player status in GUI
    def update_status():
        for player, stats in players.items():
            label = stats['label']
            status = "Eliminated" if stats['status'] == 'eliminated' else "Alive"
            label.config(text=f"{player}: {status}")
        root.after(100, update_status)
    
    root.after(100, update_status)
    root.mainloop()

def main():
    # Step 1: Select Simulation Mode
    simulation_mode = select_simulation_mode()
    
    # Step 2: Get User Inputs Based on Mode
    params = get_user_input(simulation_mode)
    
    # Step 3: Run Simulations or Launch Interactive Mode
    if params['interactive']:
        print("\nLaunching Real-Time Interactive Game...")
        interactive_game(params)
        sys.exit()
    
    if params['betting']:
        print("\nRunning simulations with Betting and Wagering System...")
    else:
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
