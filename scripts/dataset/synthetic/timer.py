import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import uuid

# --- Habit Data Model ---
class Habit:
    """Represents a single daily habit."""
    def __init__(self, name, is_checked=False):
        # Generate a unique ID for easier data manipulation
        self.id = str(uuid.uuid4())
        self.name = name
        self.is_checked = is_checked

# --- Main Application ---
class HabitTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Daily Habit Tracker")
        self.root.geometry("450x600")
        self.habits = []  # List to store Habit objects
        
        # Apply a basic style
        style = ttk.Style()
        style.configure("TFrame", background="#f0f4f8")
        style.configure("TLabel", background="#f0f4f8", font=('Arial', 10))
        style.configure("TButton", font=('Arial', 9, 'bold'))
        style.configure("Habit.TLabel", font=('Arial', 12))
        
        self.setup_ui()
        
        # Load some initial dummy data
        self.habits.append(Habit("Drink 8 glasses of water", True))
        self.habits.append(Habit("Read for 30 minutes"))
        self.habits.append(Habit("Go for a 15-minute walk"))

        self.refresh_habit_list()

    def setup_ui(self):
        # --- Input Section (Top) ---
        input_frame = ttk.Frame(self.root, padding="10 10 10 0")
        input_frame.pack(fill='x')

        ttk.Label(input_frame, text="New Habit Name:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.habit_name_entry = ttk.Entry(input_frame, width=30)
        self.habit_name_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 10))
        
        add_button = ttk.Button(input_frame, text="Add Habit", command=self.add_habit)
        add_button.pack(side=tk.LEFT)
        
        # Allow hitting Enter to add the habit
        self.root.bind('<Return>', lambda event: self.add_habit())

        # --- Habit List Section (Middle) ---
        
        # Use a canvas and a frame inside it to make the content scrollable
        self.canvas = tk.Canvas(self.root, background="#f0f4f8", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.scrollbar.pack(side="right", fill="y", pady=10)

    def add_habit(self):
        """Adds a new habit based on the entry field content."""
        name = self.habit_name_entry.get().strip()
        if name:
            new_habit = Habit(name)
            self.habits.append(new_habit)
            self.habit_name_entry.delete(0, tk.END) # Clear the input field
            self.refresh_habit_list()
        else:
            messagebox.showwarning("Input Error", "Please enter a name for the habit.")

    def delete_habit(self, habit_id):
        """Deletes a habit by its ID."""
        # Filter out the habit with the matching ID
        self.habits = [h for h in self.habits if h.id != habit_id]
        self.refresh_habit_list()

    def edit_habit(self, habit_id):
        """Opens a dialog to edit the habit's name."""
        habit = next((h for h in self.habits if h.id == habit_id), None)
        if not habit:
            return

        # Simpledialog is great for quick text input
        new_name = simpledialog.askstring(
            "Edit Habit", 
            f"Enter new name for: '{habit.name}'",
            initialvalue=habit.name
        )
        
        if new_name and new_name.strip() and new_name != habit.name:
            habit.name = new_name.strip()
            self.refresh_habit_list()

    def toggle_habit(self, habit_id):
        """Toggles the 'is_checked' status of a habit by its ID."""
        habit = next((h for h in self.habits if h.id == habit_id), None)
        if habit:
            habit.is_checked = not habit.is_checked
            self.refresh_habit_list()

    def refresh_habit_list(self):
        """
        Clears the scrollable frame and rebuilds the list of habits from the self.habits data.
        This is called after every data change (add, check, edit, delete).
        """
        # Destroy all existing widgets in the scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Sort unchecked habits first, then checked ones
        self.habits.sort(key=lambda h: h.is_checked)
        
        if not self.habits:
            ttk.Label(
                self.scrollable_frame, 
                text="No habits yet! Add one above to get started.",
                padding="20"
            ).pack(fill='x')
            
        for i, habit in enumerate(self.habits):
            # Create a frame for each habit entry
            habit_row = ttk.Frame(self.scrollable_frame, padding="10", relief="flat")
            habit_row.pack(fill='x', pady=2, padx=5)

            # Checkbox to toggle the completion status
            check_state = tk.BooleanVar(value=habit.is_checked)
            check_button = ttk.Checkbutton(
                habit_row, 
                variable=check_state,
                command=lambda h_id=habit.id: self.toggle_habit(h_id)
            )
            check_button.pack(side=tk.LEFT, padx=(0, 10))

            # Habit Name Label
            text_style = 'Habit.TLabel'
            if habit.is_checked:
                # Add strikethrough effect for completed habits
                text_style = 'Completed.TLabel'
                style = ttk.Style()
                style.configure(text_style, font=('Arial', 12, 'overstrike'), foreground='#6c757d', background="#f0f4f8")

            habit_label = ttk.Label(
                habit_row, 
                text=habit.name, 
                style=text_style
            )
            habit_label.pack(side=tk.LEFT, fill='x', expand=True, anchor='w')

            # Edit Button
            edit_button = ttk.Button(
                habit_row, 
                text="✏️ Edit",
                command=lambda h_id=habit.id: self.edit_habit(h_id)
            )
            edit_button.pack(side=tk.RIGHT, padx=5)

            # Delete Button
            delete_button = ttk.Button(
                habit_row, 
                text="❌ Delete", 
                command=lambda h_id=habit.id: self.delete_habit(h_id)
            )
            delete_button.pack(side=tk.RIGHT)
            
            # Add a separator line for visual clarity
            ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x')
        
        # Update the scroll region after adding all widgets
        self.scrollable_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = HabitTrackerApp(root)
    root.mainloop()