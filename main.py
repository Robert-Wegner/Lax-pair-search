#!/usr/bin/env python3

import os
import json
import argparse
import subprocess
import sys
from openai import OpenAI
import traceback
from sympy import Symbol, symbols, simplify
from symbolic import (
    # We import anything we need to check the Lax pairs
    check_Lax_pair
    # plus any PDE or Lax checking routines you might have
    # or anything else from symbolic.py
)

# Initialize OpenAI client
client = OpenAI()

# Paths for storing attempts
SUCCESS_PATH = "found_lax_pairs.json"
FAILURE_PATH = "attempted_lax_pairs.json"

# Use a reasoning-based model name, e.g. "gpt-4o" or "gpt-4o-reasoning-beta"
REASONING_MODEL = "o1-mini"

def load_pairs(filepath):
    """Helper to load pairs from JSON."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return json.load(f)

def save_pairs(filepath, pairs):
    """Helper to save pairs to JSON."""
    with open(filepath, "w") as f:
        json.dump(pairs, f, indent=2)

def propose_new_lax_pairs(context, max_new=1):
    """
    Call OpenAI's reasoning model with our existing context and request
    new potential Lax pairs. Return them as textual code or structured data.
    """
    system_message = (
        "You are a mathematical assistant specialized in integrable PDEs. "
        "Your task is to find new Lax pairs which yield a PDE when plugged into the Lax equation d/dt L + [L, P] = 0."
        "This means that after simplification the left hand side will be a multiplication operator by some function, and this function being zero is the PDE we obtain."
        "Ideally this is an interesting nonlinear PDE."
        "You will provide in your answer the following:"
        "The operators L and P, any differentiable symbols (functions) you have used, "
        "any non differentiable symbols (domain directions),"
        "and any constants you have used."
        "You shall write these operators in sympy code."
        "Here are an examples:"
        "1. The NLS Lax pair:"
        "L = I * Matrix([[D(x), - q], [p, - D(x)]])"
        "P = I * Matrix([[2 * D(x, x) - q * p, - q * D(x) - D(x) * q], [p * D(x) + D(x) * p, - 2 * D(x, x) + q * p]])"
        "2. The KdV Lax pair:"
        "L = - D(x, x) + u"
        "P = - 4 * D(x, x, x) + 3 * (D(x) * u - u * D(x)) + 6 * u * D(x)"
        "Note that L should be self-adjoint and P should be skew-adjoint."
        "As you can see a custom library provides you with the class D for differentiation." 
        "Write D(x), D(x, x), D(x, y), D(z) for the corresponding formal differentiation operators."
        "You have only x, y and z, t available as direction."
        "Use single lowercase letters for all other symbols."
        "For functions depending on x, y, z and t you can use a, b, f, g, h, p, q, r, s, u, v, w."
        "For constants use c, d, e, i, j, k, l, m, n, o."
        "Use constants only if you want to observe the output to then pick a correct explicit value for the constant."
        "Do not write u_x or something like that for differentiation. Only use D(x) and the differentiable functions."
        "For example D(x) * q - q - D(x) is how you represent q_x (the operator o fmultiplication by q_x)."
        "Don't try to use too many functions - one or two should be difficult enough. Use common typical function variables with preference, like u, v, w, q, p, f, g, h."
        "In fact, for now I want you to EXCLUVIELY work with the functions u, q and p and the direction x. Use only u or only q and p."
        "Do not needlessly receate the same Lax pair but with different variables again and again. Your goal is to find genuinely new Lax pairs. DO not resubmit a previous Lax pair."
        "Since you are using sympy you have access to the standard sympy constants/functions like pi, I, exp, log, sqrt etc."
        "Complex conjugation is not supported. You can not use Sympy's conjugate() function."
        # "For now, please try to find a variation of this idea which works:"
        # "L_code = Matrix([[D(x) + u, q], [p, -D(x) - u]])"
        # "P_code = Matrix([[D(x, x) + u**2 + p*q, q*D(x) + u*q], [p*D(x) - u*p, -D(x, x) - u**2 - p*q]])"
        # "equation_string = Matrix([[-2*(\\tilde{f})_{x}*(u)_{x}/\\tilde{f} + (q)_{x}*p + (u)_{t} - (u)_{xx} + 2*(u)_{x}*u - 2*p*q*u, -(\\tilde{f})_{x}*(q)_{x}/\\tilde{f} + 4*(\\tilde{f})_{x}*q*u/\\tilde{f} + (q)_{t} - (q)_{xx} + (q)_{x}*u + 2*(u)_{x}*q - 2*p*q**2], [(\\tilde{f})_{x}*(p)_{x}/\\tilde{f} + (p)_{t} + (p)_{xx} + (p)_{x}*u + 2*p**2*q + 4*p*u**2, -2*(\\tilde{f})_{x}*(u)_{x}/\\tilde{f} + (p)_{x}*q - (u)_{t} - (u)_{xx} + 2*(u)_{x}*u + 2*p*q*u]])"
        "Lastly, keep it simple. Do not add too many variables and terms."
        "Always strive to find Lax pairs SUBSTANTIALLY different from the previous ones, not just a renaming of variables or an added constant."
    )
    
    # Build the user message (with context)
    user_message = system_message + f"""
Here is the context of previously found Lax pairs (successful) and attempts (failed).
SUCCESSFUL LAX PAIRS:
{json.dumps(context['successes'], indent=2)}

FAILED LAX PAIRS:
{json.dumps(context['failures'], indent=2)}

Please propose up to {max_new} new Lax pair(s) (L, P). 
Return your proposal in JSON with the following structure:
[{{"L_code": "...", "P_code": "..."}}, ...]
Example: [{{"L_code": "I * Matrix([[D(x), - q], [p, - D(x)]])", "P_code": "I * Matrix([[2 * D(x, x) - q * p, - q * D(x) - D(x) * q], [p * D(x) + D(x) * p, - 2 * D(x, x) + q * p]])"}}]
Make sure that your reply only consists of this JSON in a codebox and nothing else. 
If you add other text like remarks, comments or explanations, my script will crash.
Thank you.
"""
    
    try:
        # IMPORTANT: use openai.chat.create (not ChatCompletion.create)
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                #{"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            # If your chosen model supports these parameters, you can include them:
            #temperature=0.7,
        )

        # The response text is in:
        reply = response.choices[0].message.content

        import re

        # Use a regex to find the first code block delimited by triple backticks
        match = re.search(r"```(?:json)?\s*(.*?)```", reply, re.DOTALL)
        if match:
            json_text = match.group(1).strip()  # Extract just the JSON part
            try:
                new_pairs = json.loads(json_text)
            except json.JSONDecodeError:
                print("Could not parse the extracted text as JSON. Raw text:\n", json_text)
                return []
        else:
            print("No code block found in the response!")
            return []

        # Ensure it's a list of dicts with "L_code" and "P_code"
        valid_structs = []
        for item in new_pairs:
            if "L_code" in item and "P_code" in item:
                valid_structs.append(item)
            else:
                print("Skipping invalid item structure:", item)
        return valid_structs
    
    except Exception as error:
        print("OpenAI API Error:", error)
        traceback.print_exc()
        return []

def agentic_search_for_lax_pairs():
    """
    Main loop:
      1) Load successes/failures
      2) Send them to the model
      3) Attempt to parse and check new pairs
      4) Record results
      5) Possibly repeat
    """
    successes = load_pairs(SUCCESS_PATH)
    failures = load_pairs(FAILURE_PATH)
    
    filtered_successes = [
        {"code": entry["code"], "equation_string": entry["equation_string"]}
        for entry in successes
        if "code" in entry and "equation_string" in entry
    ]
    
    filtered_failures = [
        {"code": entry["code"], "equation_string": entry["equation_string"]}
        for entry in failures
        if "code" in entry and "equation_string" in entry
    ]
    
    # This context is what we feed back to the model:
    context = {
        "successes": filtered_successes,
        "failures": filtered_failures
    }
    # We'll just do one round here. 
    # In practice, you might do multiple iterations or a while True loop.
    new_proposals = propose_new_lax_pairs(context, max_new=4)
    
    print("New proposals from the model:", new_proposals)
    succeeded = False
    for proposal in new_proposals:
        L_code = proposal["L_code"]
        P_code = proposal["P_code"]
        
        # Safely evaluate code in Python. For real security, you'd sanitize or run in a sandbox.
        # Here we do a naive example using `eval`.
        try:
            # Now check:
            is_Lax_pair, Lax_equation, Lax_equation_latex, L_latex, P_latex = check_Lax_pair(L_code, P_code)
            if is_Lax_pair:
                print("SUCCESS! We found a valid Lax pair.")
                succeeded = True
                successes.append({
                    "code": proposal,
                    "equation_string": str(Lax_equation),
                    "equation_latex":  Lax_equation_latex,
                    "L_latex": L_latex,
                    "P_latex": P_latex,
                })
            else:
                print("FAIL! The proposed Lax pair is not valid.")
                failures.append({
                    "code": proposal,
                    "equation_string": str(Lax_equation),
                    "equation_latex":  Lax_equation_latex,
                    "L_latex": L_latex,
                    "P_latex": P_latex,
                })
        except Exception as e:
            print("Exception while trying to evaluate or check the proposal:", e)
            failures.append(proposal)
    
    # Save updated lists
    save_pairs(SUCCESS_PATH, successes)
    save_pairs(FAILURE_PATH, failures)
    
    print("=== SUMMARY ===")
    print(f"Successful Lax Pairs so far: {len(successes)}")
    print(f"Failed attempts so far:       {len(failures)}")

    if succeeded:
        render_successes()

def clear_attempts():
    """Clears the failed attempts file."""
    if os.path.exists(FAILURE_PATH):
        os.remove(FAILURE_PATH)
        print(f"Cleared failed attempts ({FAILURE_PATH}).")
    else:
        print(f"No failed attempts to clear ({FAILURE_PATH} does not exist).")

def clear_successes():
    """Clears the successes file after user confirmation."""
    if os.path.exists(SUCCESS_PATH):
        confirmation = input(f"Are you sure you want to clear all successes in {SUCCESS_PATH}? (yes/no): ")
        if confirmation.lower() in ['yes', 'y']:
            os.remove(SUCCESS_PATH)
            print(f"Cleared successful Lax pairs ({SUCCESS_PATH}).")
        else:
            print("Clear successes operation cancelled.")
    else:
        print(f"No successful Lax pairs to clear ({SUCCESS_PATH} does not exist).")

def render_attempts():
    """Renders the successful Lax pairs and their equations into a PDF using LaTeX."""
    failures = load_pairs(FAILURE_PATH)
    if not failures:
        print("No successful Lax pairs to render.")
        return
    render(failures)

def render_successes():
    """Renders the successful Lax pairs and their equations into a PDF using LaTeX."""
    successes = load_pairs(SUCCESS_PATH)
    if not successes:
        print("No successful Lax pairs to render.")
        return
    render(successes)
    
def render(entries):
    # Build LaTeX preamble and start of document
    latex_content = r"""
\documentclass{article}
\usepackage{amsmath,amssymb}
\usepackage{geometry}

\geometry{a4paper, margin=1in}

\begin{document}

\title{Successful Lax Pairs and Equations}
\author{Generated by Agentic Lax Pair Finder}
\date{\today}
\maketitle

\begin{enumerate}
"""
    # Append each success entry
    for idx, entry in enumerate(entries, start=1):
        try:
            L_expr = entry['L_latex']
            P_expr = entry['P_latex']
            eq_expr = entry['equation_latex']
            latex_content += f"""
                \\item
                \\begin{{align}}
                L_{{{idx}}} &= {L_expr} \\\\
                P_{{{idx}}} &= {P_expr}
                \\end{{align}}
                
                \\begin{{equation}}
                {eq_expr}
                \\end{{equation}}
            """
        except:
            L_expr = ""
            P_expr = ""
            eq_expr = ""

        # Notice the double braces around idx: L_{{{idx}}} so that
        # the subscript is interpreted as {1}, {2}, etc.

    # Close the enumerate environment and the document
    latex_content += r"""
\end{enumerate}

\end{document}
"""

    # Write LaTeX content to a .tex file
    tex_filename = "lax_pairs.tex"
    with open(tex_filename, "w", encoding="utf-8") as tex_file:
        tex_file.write(latex_content)

    print(f"LaTeX document created: {tex_filename}")

    # Compile LaTeX to PDF using pdflatex
    try:
        subprocess.run(["pdflatex", tex_filename], check=True)
        print("PDF generated successfully.")
    except subprocess.CalledProcessError as e:
        print("Error occurred while compiling LaTeX document:", e)
        print("Please ensure that pdflatex is installed and available in your PATH.")
        return

    # Optionally, clean up auxiliary files generated by LaTeX
    for ext in ['aux', 'log']:
        aux_file = f"lax_pairs.{ext}"
        if os.path.exists(aux_file):
            os.remove(aux_file)
            print(f"Removed auxiliary file: {aux_file}")

import argparse


def main():
    print("Welcome to the Agentic Lax Pair Finder Interactive Shell!")
    print("Type 'help' to see available commands.\n")

    while True:
        try:
            user_input = input("Enter command: ").strip()
            if not user_input:
                continue  # Skip empty input

            parts = user_input.split()
            command = parts[0].lower()

            if command == "search":
                if len(parts) == 2:
                    try:
                        n = int(parts[1])
                        for i in range(n):
                            print(f"Search iteration {i+1} of {n}:")
                            agentic_search_for_lax_pairs()
                    except ValueError:
                        print("Error: The 'search' command requires an integer argument. Usage: search [n]")
                elif len(parts) == 1:
                    agentic_search_for_lax_pairs()
                else:
                    print("Error: Invalid usage of 'search'. Usage: search [n]")

            elif command == "clear_attempts":
                clear_attempts()

            elif command == "clear_successes":
                clear_successes()

            elif command == "render_attempts":
                render_attempts()
            
            elif command == "render_successes":
                render_successes()

            elif command == "help":
                print("\nAvailable commands:")
                print("  search [n]         : Start the agent to search for new Lax pairs. Optionally run it 'n' times.")
                print("  clear_attempts     : Clear the failed attempts file.")
                print("  clear_successes    : Clear the successful Lax pairs file.")
                print("  render_attempts    : Render successful Lax pairs into a PDF document using LaTeX.")
                print("  render_successes    : Render failed attempts at Lax pairs into a PDF document using LaTeX.")
                print("  exit               : Exit the interactive shell.\n")

            elif command == "exit":
                print("Exiting the Agentic Lax Pair Finder. Goodbye!")
                break

            else:
                print(f"Unknown command: '{command}'. Type 'help' to see available commands.")

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Exiting.")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
