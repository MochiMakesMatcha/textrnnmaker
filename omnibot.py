import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pickle
import json
from threading import Thread, Event
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import time
import sys

# Tooltip class for hover help
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.show_tip)
        widget.bind('<Leave>', self.hide_tip)

    def show_tip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Arial", 10))
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class PrintLogger:
    def __init__(self, log_callback):
        self.log_callback = log_callback

    def write(self, text):
        if text.strip():
            self.log_callback(text.strip())

    def flush(self):
        pass

class TextGenerator:
    def __init__(self, mode='character'):
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.sequence_length = None
        self.total_tokens = None
        self.stop_flag = False

        # Track training history for graphs
        self.training_history = {
            'accuracy': [],
            'loss': [],
            'val_accuracy': [],
            'val_loss': []
        }

    # ===================== DATA =====================

    def prepare_sequences(self, text, sequence_length, validation_split=0.2):
        if self.mode == 'word':
            return self._prepare_word_sequences(text, sequence_length, validation_split)
        else:
            return self._prepare_character_sequences(text, sequence_length, validation_split)

    def _prepare_word_sequences(self, text, sequence_length, validation_split):
        self.tokenizer = Tokenizer(filters='')
        self.tokenizer.fit_on_texts([text])

        self.total_tokens = len(self.tokenizer.word_index) + 1
        self.index_to_word = {v: k for k, v in self.tokenizer.word_index.items()}

        words = text.lower().split()
        sequences = []

        for i in range(sequence_length, len(words)):
            seq = words[i-sequence_length:i+1]
            sequences.append(seq)

        sequences = self.tokenizer.texts_to_sequences([' '.join(seq) for seq in sequences])
        sequences = pad_sequences(sequences, maxlen=sequence_length + 1, padding='pre')

        X = sequences[:, :-1]
        y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=self.total_tokens)

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        split = int(len(X) * (1 - validation_split))
        return (X[:split], y[:split]), (X[split:], y[split:])

    def _prepare_character_sequences(self, text, sequence_length, validation_split):
        text = text.lower()

        chars = sorted(list(set(text)))
        self.total_tokens = len(chars) + 1

        self.char_to_int = {c: i+1 for i, c in enumerate(chars)}
        self.int_to_char = {i+1: c for i, c in enumerate(chars)}

        sequences = []
        for i in range(sequence_length, len(text)):
            seq = text[i-sequence_length:i+1]
            sequences.append([self.char_to_int.get(c, 0) for c in seq])

        sequences = np.array(sequences)

        X = sequences[:, :-1]
        y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=self.total_tokens)

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        split = int(len(X) * (1 - validation_split))
        return (X[:split], y[:split]), (X[split:], y[split:])

    # ===================== MODEL =====================

    def build_model(self, embedding_dim=64, lstm_units=128, dropout_rate=0.3):
        model = Sequential([
            Embedding(self.total_tokens, embedding_dim),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dense(self.total_tokens, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
            metrics=['accuracy']
        )

        self.model = model
        return model

    # ===================== TRAIN =====================

    def train(self, text, sequence_length, epochs, batch_size,
              validation_split=0.2,
              callback=None,
              embedding_dim=64,
              lstm_units=128,
              dropout_rate=0.3,
              use_early_stopping=True,
              patience=5):

        self.stop_flag = False
        self.sequence_length = sequence_length

        self.training_history = {
            'accuracy': [],
            'loss': [],
            'val_accuracy': [],
            'val_loss': []
        }

        (X_train, y_train), (X_val, y_val) = self.prepare_sequences(
            text, sequence_length, validation_split
        )

        self.build_model(embedding_dim, lstm_units, dropout_rate)

        callbacks = []

        # Early stopping
        if use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ))

        # LR scheduler (helps converge faster)
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-5
        ))

        # Custom callback for GUI updates + stop button
        class TrainingCallback(Callback):
            def __init__(self, outer):
                self.outer = outer

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}

                self.outer.training_history['accuracy'].append(logs.get('accuracy', 0))
                self.outer.training_history['loss'].append(logs.get('loss', 0))
                self.outer.training_history['val_accuracy'].append(logs.get('val_accuracy', 0))
                self.outer.training_history['val_loss'].append(logs.get('val_loss', 0))

                if callback:
                    callback(
                        epoch + 1,
                        logs.get('loss', 0),
                        logs.get('accuracy', 0),
                        logs.get('val_loss', 0),
                        logs.get('val_accuracy', 0)
                    )

                if self.outer.stop_flag:
                    self.model.stop_training = True

        callbacks.append(TrainingCallback(self))

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def stop_training(self):
        self.stop_flag = True

    # ===================== GENERATION =====================

    def _sample(self, preds, temperature):
        preds = np.log(np.maximum(preds, 1e-8)) / temperature
        preds = np.exp(preds)
        preds /= np.sum(preds)
        return np.random.choice(len(preds), p=preds)

    def generate_text(self, prompt, num_predictions=100, temperature=1.0):
        if self.mode == 'word':
            return self._generate_words(prompt, num_predictions, temperature)
        else:
            return self._generate_chars(prompt, num_predictions, temperature)

    def _generate_chars(self, prompt, length, temperature):
        result = list(prompt.lower())

        for _ in range(length):
            seq = result[-self.sequence_length:]
            seq = [self.char_to_int.get(c, 0) for c in seq]
            seq = pad_sequences([seq], maxlen=self.sequence_length, padding='pre')

            preds = self.model(seq, training=False).numpy()[0]
            idx = self._sample(preds, temperature)

            result.append(self.int_to_char.get(idx, '?'))

        return ''.join(result)

    def _generate_words(self, prompt, length, temperature):
        result = prompt.lower().split()

        for _ in range(length):
            seq = result[-self.sequence_length:]
            seq = self.tokenizer.texts_to_sequences([' '.join(seq)])[0]
            seq = pad_sequences([seq], maxlen=self.sequence_length, padding='pre')

            preds = self.model(seq, training=False).numpy()[0]
            idx = self._sample(preds, temperature)

            word = self.index_to_word.get(idx)
            if not word:
                break

            result.append(word)

        return ' '.join(result)

    # ===================== STATS =====================

    def get_model_stats(self):
        if not self.training_history['accuracy']:
            return None

        return {
            'final_accuracy': self.training_history['accuracy'][-1],
            'final_loss': self.training_history['loss'][-1],
            'best_accuracy': max(self.training_history['accuracy']),
            'best_val_accuracy': max(self.training_history['val_accuracy']),
            'epochs_completed': len(self.training_history['accuracy'])
        }

    # Add these methods to the TextGenerator class:

    def save_model(self, filepath):
        """Save model, tokenizer, and configuration"""
        try:
            # Save model
            self.model.save(f"{filepath}_model.h5")

            # Save tokenizer/char mappings
            if self.mode == 'word':
                with open(f"{filepath}_tokenizer.pkl", 'wb') as f:
                    pickle.dump(self.tokenizer, f)
            else:
                with open(f"{filepath}_char_mappings.pkl", 'wb') as f:
                    pickle.dump({
                        'char_to_int': self.char_to_int,
                        'int_to_char': self.int_to_char,
                        'total_tokens': self.total_tokens
                    }, f)

            # Save config
            config = {
                'mode': self.mode,
                'sequence_length': self.sequence_length,
                'total_tokens': self.total_tokens,
                'training_history': self.training_history
            }
            with open(f"{filepath}_config.json", 'w') as f:
                json.dump(config, f)

            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def load_model(self, filepath):
        """Load model, tokenizer, and configuration"""
        try:
            # Load config first
            with open(f"{filepath}_config.json", 'r') as f:
                config = json.load(f)

            self.mode = config['mode']
            self.sequence_length = config['sequence_length']
            self.total_tokens = config['total_tokens']
            self.training_history = config['training_history']

            # Load model
            self.model = load_model(f"{filepath}_model.h5")

            # Load tokenizer/mappings
            if self.mode == 'word':
                with open(f"{filepath}_tokenizer.pkl", 'rb') as f:
                    self.tokenizer = pickle.load(f)
                self.index_to_word = {v: k for k, v in self.tokenizer.word_index.items()}
            else:
                with open(f"{filepath}_char_mappings.pkl", 'rb') as f:
                    mappings = pickle.load(f)
                self.char_to_int = mappings['char_to_int']
                self.int_to_char = mappings['int_to_char']

            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

    # Fix the ToolTip class:
    class ToolTip:
        def __init__(self, widget, text):
            self.widget = widget
            self.text = text
            self.tip_window = None
            widget.bind('<Enter>', self.show_tip)
            widget.bind('<Leave>', self.hide_tip)

        def show_tip(self, event=None):
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25

            self.tip_window = tk.Toplevel(self.widget)  # Store as instance variable
            self.tip_window.wm_overrideredirect(True)
            self.tip_window.wm_geometry(f"+{x}+{y}")

            label = tk.Label(self.tip_window, text=self.text, justify=tk.LEFT,
                            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                            font=("Arial", 10))
            label.pack()

        def hide_tip(self, event=None):
            if self.tip_window:
                self.tip_window.destroy()
            self.tip_window = None

class LogHandler:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.log_queue = deque(maxlen=1000)

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}\n"

        self.log_queue.append(formatted_message)

        def update_text():
            self.text_widget.insert(tk.END, formatted_message)
            self.text_widget.see(tk.END)

            if level == "ERROR":
                self.text_widget.tag_add("error", f"end-2l", f"end-1l")
                self.text_widget.tag_config("error", foreground="red")
            elif level == "WARNING":
                self.text_widget.tag_add("warning", f"end-2l", f"end-1l")
                self.text_widget.tag_config("warning", foreground="orange")
            elif level == "SUCCESS":
                self.text_widget.tag_add("success", f"end-2l", f"end-1l")
                self.text_widget.tag_config("success", foreground="green")

        if self.text_widget.winfo_exists():
            self.text_widget.after(0, update_text)

    def clear(self):
        self.text_widget.delete(1.0, tk.END)
        self.log_queue.clear()

class TextGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Generator - Word & Character Level")
        self.root.geometry("1200x800")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Initialize all variables BEFORE setup_ui
        self.generator = None
        self.training = False
        self.training_thread = None
        self.log_handler = None

        # Initialize all IntVar and DoubleVar objects
        self.context_size = tk.IntVar(value=5)
        self.embedding_dim = tk.IntVar(value=100)
        self.lstm_units = tk.IntVar(value=256)
        self.dropout_rate = tk.DoubleVar(value=0.2)
        self.epochs = tk.IntVar(value=50)
        self.batch_size = tk.IntVar(value=64)
        self.use_early_stopping = tk.BooleanVar(value=True)
        self.patience = tk.IntVar(value=5)
        self.mode_var = tk.StringVar(value="word")
        self.length_to_gen = tk.IntVar(value=50)
        self.temperature = tk.DoubleVar(value=1.0)

        # Now setup UI
        self.setup_ui()
        self.log_handler = LogHandler(self.log_text)

        sys.stdout = PrintLogger(lambda msg: self.log_handler.log(msg, "INFO"))

        self.log_handler.log("Application started. Select mode and load/create model.")

        # Set reasonable defaults for ~7k characters
        self.set_reasonable_defaults()

    def set_reasonable_defaults(self):
        """Set reasonable defaults for ~7k character training text"""
        # For character mode with ~7k chars
        self.context_size.set(30)  # Look at last 30 characters
        self.embedding_dim.set(64)  # Lower dimension for smaller dataset
        self.lstm_units.set(128)    # Fewer units to prevent overfitting
        self.dropout_rate.set(0.3)   # Higher dropout for regularization
        self.epochs.set(100)         # More epochs since smaller model trains faster
        self.batch_size.set(32)      # Smaller batch size for better generalization
        self.use_early_stopping.set(True)
        self.patience.set(10)        # More patience for slower learning

        # Log the defaults
        self.log_handler.log("Reasonable defaults loaded for ~7k character training text", "INFO")
        self.log_handler.log("Context size: 30 chars, Embedding: 64, LSTM: 128, Dropout: 0.3", "INFO")

    def add_tooltip(self, widget, text):
        """Add a tooltip to a widget"""
        ToolTip(widget, text)

    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Create PanedWindow for resizable sections
        paned = ttk.PanedWindow(main_container, orient="vertical")
        paned.pack(fill="both", expand=True)

        # Top section - Training and Configuration
        top_frame = ttk.Frame(paned)
        paned.add(top_frame, weight=2)

        # Split top section into left and right
        top_paned = ttk.PanedWindow(top_frame, orient="horizontal")
        top_paned.pack(fill="both", expand=True)

        # Left side - Training
        train_frame = ttk.LabelFrame(top_paned, text="Training Configuration", padding=10)
        top_paned.add(train_frame, weight=1)

        # Mode selection at top
        mode_frame = ttk.LabelFrame(train_frame, text="Generation Mode", padding=10)
        mode_frame.grid(row=0, column=0, columnspan=5, sticky="ew", pady=5)

        word_radio = ttk.Radiobutton(mode_frame, text="Word Level", variable=self.mode_var, value="word",
                       command=self.switch_mode)
        word_radio.pack(side="left", padx=10)
        self.add_tooltip(word_radio, "Predicts whole words. Best for coherent sentences.\nNeeds ~10,000+ words of training data.")

        char_radio = ttk.Radiobutton(mode_frame, text="Character Level", variable=self.mode_var, value="character",
                       command=self.switch_mode)
        char_radio.pack(side="left", padx=10)
        self.add_tooltip(char_radio, "Predicts one character at a time.\nWorks with smaller datasets (like ~7k chars).\nCan learn spelling and create novel words.")

        # Mode info label
        self.mode_info_label = ttk.Label(mode_frame, text="", foreground="blue")
        self.mode_info_label.pack(side="left", padx=20)

        # Training text
        ttk.Label(train_frame, text="Training Text:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky="w", pady=5)
        self.train_text = tk.Text(train_frame, height=12, wrap=tk.WORD)
        self.train_text.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=5)

        # Scrollbar for training text
        train_scroll = ttk.Scrollbar(train_frame, orient="vertical", command=self.train_text.yview)
        train_scroll.grid(row=2, column=5, sticky="ns")
        self.train_text.configure(yscrollcommand=train_scroll.set)

        # Button frame
        btn_frame = ttk.Frame(train_frame)
        btn_frame.grid(row=3, column=0, columnspan=5, pady=10)

        load_example_btn = ttk.Button(btn_frame, text="Load Example Text", command=self.load_example)
        load_example_btn.pack(side="left", padx=5)
        self.add_tooltip(load_example_btn, "Loads a ~500 word example text.\nGood for quick testing.")

        clear_btn = ttk.Button(btn_frame, text="Clear Text", command=lambda: self.train_text.delete(1.0, tk.END))
        clear_btn.pack(side="left", padx=5)
        self.add_tooltip(clear_btn, "Clears all training text.")

        # Parameters frame
        params_frame = ttk.LabelFrame(train_frame, text="Training Parameters", padding=10)
        params_frame.grid(row=4, column=0, columnspan=5, sticky="ew", pady=10)

        # Parameter grid
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill="x")

        # Row 1
        # Context Size
        ttk.Label(params_grid, text="Context Size:").grid(row=0, column=0, sticky="w", pady=5, padx=5)
        self.context_spinbox = ttk.Spinbox(params_grid, from_=2, to=20, textvariable=self.context_size, width=10)
        self.context_spinbox.grid(row=0, column=1, sticky="w", pady=5, padx=5)
        self.add_tooltip(self.context_spinbox,
                        "Word mode: How many previous words to consider (2-20)\n"
                        "Character mode: How many previous characters (10-100)\n"
                        "Larger = more context but slower training\n"
                        "Recommended: 5-10 words OR 30-50 characters")

        self.context_note = ttk.Label(params_grid, text="(words)", foreground="gray")
        self.context_note.grid(row=0, column=2, sticky="w", pady=5, padx=5)

        # Embedding Dimension
        ttk.Label(params_grid, text="Embedding Dim:").grid(row=0, column=3, sticky="w", pady=5, padx=5)
        embedding_spin = ttk.Spinbox(params_grid, from_=16, to=300, textvariable=self.embedding_dim, width=10)
        embedding_spin.grid(row=0, column=4, sticky="w", pady=5, padx=5)
        self.add_tooltip(embedding_spin,
                        "How complex the word/character representations are.\n"
                        "Higher = better patterns but more memory.\n"
                        "~7k chars: 64 is good. ~50k+ words: 200+ is better.")

        # Row 2
        # LSTM Units
        ttk.Label(params_grid, text="LSTM Units:").grid(row=1, column=0, sticky="w", pady=5, padx=5)
        lstm_spin = ttk.Spinbox(params_grid, from_=32, to=512, textvariable=self.lstm_units, width=10)
        lstm_spin.grid(row=1, column=1, sticky="w", pady=5, padx=5)
        self.add_tooltip(lstm_spin,
                        "Number of memory cells in the network.\n"
                        "Higher = more learning capacity but slower.\n"
                        "Small data (7k chars): 64-128\n"
                        "Large data (100k+ words): 256-512")

        # Dropout Rate
        ttk.Label(params_grid, text="Dropout Rate:").grid(row=1, column=3, sticky="w", pady=5, padx=5)
        dropout_spin = ttk.Spinbox(params_grid, from_=0.1, to=0.5, increment=0.05, textvariable=self.dropout_rate, width=10)
        dropout_spin.grid(row=1, column=4, sticky="w", pady=5, padx=5)
        self.add_tooltip(dropout_spin,
                        "Randomly drops neurons to prevent overfitting.\n"
                        "Higher = more regularization.\n"
                        "Small data (7k chars): 0.3-0.5\n"
                        "Large data: 0.1-0.2")

        # Row 3
        # Epochs
        ttk.Label(params_grid, text="Epochs:").grid(row=2, column=0, sticky="w", pady=5, padx=5)
        epochs_spin = ttk.Spinbox(params_grid, from_=10, to=500, textvariable=self.epochs, width=10)
        epochs_spin.grid(row=2, column=1, sticky="w", pady=5, padx=5)
        self.add_tooltip(epochs_spin,
                        "Number of complete passes through the training data.\n"
                        "More = better learning but diminishing returns.\n"
                        "Stop when validation metrics plateau.")

        # Batch Size
        ttk.Label(params_grid, text="Batch Size:").grid(row=2, column=3, sticky="w", pady=5, padx=5)
        batch_spin = ttk.Spinbox(params_grid, from_=8, to=256, textvariable=self.batch_size, width=10)
        batch_spin.grid(row=2, column=4, sticky="w", pady=5, padx=5)
        self.add_tooltip(batch_spin,
                        "Number of samples before updating weights.\n"
                        "Smaller = slower but better generalization.\n"
                        "Small data: 16-32, Large data: 64-128")

        # Row 4
        # Early Stopping
        early_stop_check = ttk.Checkbutton(params_grid, text="Enable Early Stopping", variable=self.use_early_stopping)
        early_stop_check.grid(row=3, column=0, sticky="w", pady=5, padx=5)
        self.add_tooltip(early_stop_check,
                        "Stops training when validation loss stops improving.\n"
                        "Prevents overfitting and saves time.\n"
                        "Highly recommended!")

        # Patience
        ttk.Label(params_grid, text="Patience:").grid(row=3, column=3, sticky="w", pady=5, padx=5)
        patience_spin = ttk.Spinbox(params_grid, from_=1, to=30, textvariable=self.patience, width=10)
        patience_spin.grid(row=3, column=4, sticky="w", pady=5, padx=5)
        self.add_tooltip(patience_spin,
                        "How many epochs to wait for improvement before stopping.\n"
                        "Higher = trains longer even if progress is slow.\n"
                        "5-10 is usually good.")

        # Training buttons
        train_btn_frame = ttk.Frame(train_frame)
        train_btn_frame.grid(row=5, column=0, columnspan=5, pady=10)

        self.train_btn = ttk.Button(train_btn_frame, text="Start Training", command=self.start_training, width=15)
        self.train_btn.pack(side="left", padx=5)
        self.add_tooltip(self.train_btn, "Begins training the model with current settings.")

        self.stop_btn = ttk.Button(train_btn_frame, text="Stop Training", command=self.stop_training, width=15, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.add_tooltip(self.stop_btn, "Gracefully stops training after current epoch.")

        save_btn = ttk.Button(train_btn_frame, text="Save Model", command=self.save_model, width=15)
        save_btn.pack(side="left", padx=5)
        self.add_tooltip(save_btn, "Saves the trained model, tokenizer, and config.\n"
                                 "Creates 3 files: .h5, .pkl, .json")

        load_btn = ttk.Button(train_btn_frame, text="Load Model", command=self.load_model, width=15)
        load_btn.pack(side="left", padx=5)
        self.add_tooltip(load_btn, "Loads a previously saved model.\n"
                                 "Select the .h5 file - others load automatically.")

        # Progress bar
        self.progress = ttk.Progressbar(train_frame, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=5, sticky="ew", pady=5)
        self.add_tooltip(self.progress, "Shows training progress.")

        # Training status
        self.status_label = ttk.Label(train_frame, text="Ready", font=('Arial', 9))
        self.status_label.grid(row=7, column=0, columnspan=5, pady=5)

        train_frame.grid_rowconfigure(2, weight=1)
        train_frame.grid_columnconfigure(0, weight=1)

        # Right side - Stats and Visualization
        stats_frame = ttk.LabelFrame(top_paned, text="Model Statistics & Performance", padding=10)
        top_paned.add(stats_frame, weight=1)

        # Create figure for plots
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, stats_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats table
        stats_table_frame = ttk.Frame(stats_frame)
        stats_table_frame.pack(fill="x", pady=10)

        columns = ('Metric', 'Current', 'Best', 'Target')
        self.stats_tree = ttk.Treeview(stats_table_frame, columns=columns, show='headings', height=6)

        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=100)

        self.stats_tree.pack(fill="x")

        self.targets = {
            'Accuracy': 0.95,
            'Loss': 0.05,
            'Validation Accuracy': 0.90,
            'Validation Loss': 0.10
        }

        # Bottom section - Generation and Logs
        bottom_frame = ttk.Frame(paned)
        paned.add(bottom_frame, weight=1)

        bottom_paned = ttk.PanedWindow(bottom_frame, orient="horizontal")
        bottom_paned.pack(fill="both", expand=True)

        # Generation frame
        gen_frame = ttk.LabelFrame(bottom_paned, text="Text Generation", padding=10)
        bottom_paned.add(gen_frame, weight=1)

        ttk.Label(gen_frame, text="Prompt:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky="w", pady=5)
        self.prompt_text = tk.Text(gen_frame, height=3, wrap=tk.WORD)
        self.prompt_text.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        self.add_tooltip(self.prompt_text, "Enter starting text. Model will continue from here.")

        param_frame = ttk.Frame(gen_frame)
        param_frame.grid(row=2, column=0, columnspan=3, pady=10)

        ttk.Label(param_frame, text="Length to Generate:").pack(side="left", padx=5)
        length_spin = ttk.Spinbox(param_frame, from_=10, to=1000, textvariable=self.length_to_gen, width=10)
        length_spin.pack(side="left", padx=5)
        self.add_tooltip(length_spin, "How many words (or characters) to generate.")

        self.length_unit_label = ttk.Label(param_frame, text="words")
        self.length_unit_label.pack(side="left", padx=5)

        ttk.Label(param_frame, text="Temperature:").pack(side="left", padx=5)
        temp_scale = ttk.Scale(param_frame, from_=0.2, to=2.0, variable=self.temperature, orient="horizontal", length=150)
        temp_scale.pack(side="left", padx=5)
        self.add_tooltip(temp_scale,
                        "Creativity control:\n"
                        "0.2-0.5 = Very predictable, safe\n"
                        "0.6-1.0 = Balanced, creative\n"
                        "1.1-2.0 = Very creative, may be chaotic")

        self.temp_label = ttk.Label(param_frame, text="1.0")
        self.temp_label.pack(side="left", padx=5)

        generate_btn = ttk.Button(gen_frame, text="Generate Text", command=self.generate_text)
        generate_btn.grid(row=3, column=0, pady=10)
        self.add_tooltip(generate_btn, "Generates continuation of your prompt.")

        ttk.Label(gen_frame, text="Generated Text:", font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky="w", pady=5)
        self.output_text = tk.Text(gen_frame, height=8, wrap=tk.WORD)
        self.output_text.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=5)

        output_scroll = ttk.Scrollbar(gen_frame, orient="vertical", command=self.output_text.yview)
        output_scroll.grid(row=5, column=3, sticky="ns")
        self.output_text.configure(yscrollcommand=output_scroll.set)

        gen_frame.grid_rowconfigure(5, weight=1)
        gen_frame.grid_columnconfigure(0, weight=1)

        # Log frame
        log_frame = ttk.LabelFrame(bottom_paned, text="Training Log", padding=10)
        bottom_paned.add(log_frame, weight=1)

        # Add text widget for logs with scrollbar
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_text_frame, height=15, wrap=tk.WORD, font=('Courier', 9))
        self.log_text.pack(side="left", fill="both", expand=True)

        log_scroll = ttk.Scrollbar(log_text_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # Add clear log button
        clear_log_btn = ttk.Button(log_frame, text="Clear Log", command=self.clear_log)
        clear_log_btn.pack(pady=5)
        self.add_tooltip(clear_log_btn, "Clears the training log.")

        self.temperature.trace('w', lambda *args: self.temp_label.configure(text=f"{self.temperature.get():.2f}"))

        self.update_mode_info()
        self.update_plots()

    def clear_log(self):
        """Clear the log text widget"""
        self.log_text.delete(1.0, tk.END)
        self.log_handler.log("Log cleared", "INFO")

    def update_mode_info(self):
        """Update UI based on selected mode"""
        mode = self.mode_var.get()
        if mode == "word":
            self.mode_info_label.configure(text="🎯 Word mode: Predicts next WORD, needs more training data, more coherent output")
            self.context_note.configure(text="(words)")
            self.length_unit_label.configure(text="words")
            self.length_to_gen.set(50)
            self.context_spinbox.configure(from_=2, to=20)
            self.context_size.set(5)
        else:
            self.mode_info_label.configure(text="✨ Character mode: Predicts next CHARACTER, needs less data, can learn spelling and patterns")
            self.context_note.configure(text="(characters)")
            self.length_unit_label.configure(text="characters")
            self.length_to_gen.set(200)
            self.context_spinbox.configure(from_=10, to=100)
            self.context_size.set(30)  # Good default for 7k chars

    def switch_mode(self):
        """Switch between word and character modes"""
        if self.training:
            self.log_handler.log("Cannot switch mode while training!", "WARNING")
            self.mode_var.set(self.generator.mode if self.generator else "word")
            return

        self.update_mode_info()
        self.generator = TextGenerator(mode=self.mode_var.get())
        self.log_handler.log(f"Switched to {self.mode_var.get()} level mode", "INFO")
        self.status_label.configure(text=f"Ready - {self.mode_var.get()} mode")

        self.update_plots()
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

    def load_example(self):
        mode = self.mode_var.get()
        if mode == "word":
            example_text = """The quick brown fox jumps over the lazy dog. This is a simple example text for training the word-level RNN generator. The model will learn patterns and relationships between words to generate new coherent text. Context size determines how many previous words the model uses to predict the next word. The training process adjusts the model weights to minimize prediction error. After training, you can provide a prompt and the model will attempt to continue the text in a similar style. This example has about 100 words which is quite small. For better results, use at least 10,000 words of training text. The model learns from patterns, so diverse text with varied vocabulary works best."""
        else:
            example_text = """The quick brown fox jumps over the lazy dog. This demonstrates character-level generation. Each character is predicted one at a time, allowing the model to learn spelling, punctuation, and patterns at a finer granularity. Character models can generate novel words and handle out-of-vocabulary terms better than word models. For ~7000 characters, this is a good size! The model will learn the patterns in this text and be able to generate similar content. You can train on poems, code, or any text you like. Character models are great for smaller datasets because they don't need to learn a huge vocabulary. Just 7000 characters can produce surprisingly good results, especially if the text has repetitive patterns or structures."""

        self.train_text.delete(1.0, tk.END)
        self.train_text.insert(1.0, example_text)

        char_count = len(example_text)
        word_count = len(example_text.split())
        self.log_handler.log(f"Example text loaded for {mode} mode ({char_count} chars, {word_count} words)", "INFO")

    def update_training_status(self, epoch, loss, accuracy, val_loss, val_accuracy):
        """Update training progress - called from training thread via after()"""
        if self.training:
            progress_value = (epoch / self.epochs.get()) * 100
            self.progress['value'] = progress_value
            self.status_label.configure(text=f"Epoch {epoch}/{self.epochs.get()} - Loss: {loss:.4f} - Acc: {accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            self.update_stats_table(accuracy, loss, val_accuracy, val_loss)
            self.update_plots()

    def update_stats_table(self, current_acc, current_loss, current_val_acc, current_val_loss):
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        best_acc = max(self.generator.training_history['accuracy']) if self.generator.training_history['accuracy'] else current_acc
        best_loss = min(self.generator.training_history['loss']) if self.generator.training_history['loss'] else current_loss
        best_val_acc = max(self.generator.training_history['val_accuracy']) if self.generator.training_history['val_accuracy'] else current_val_acc
        best_val_loss = min(self.generator.training_history['val_loss']) if self.generator.training_history['val_loss'] else current_val_loss

        metrics = [
            ('Accuracy', f"{current_acc:.4f}", f"{best_acc:.4f}", f"{self.targets['Accuracy']:.2f}"),
            ('Loss', f"{current_loss:.4f}", f"{best_loss:.4f}", f"{self.targets['Loss']:.2f}"),
            ('Val Accuracy', f"{current_val_acc:.4f}", f"{best_val_acc:.4f}", f"{self.targets['Validation Accuracy']:.2f}"),
            ('Val Loss', f"{current_val_loss:.4f}", f"{best_val_loss:.4f}", f"{self.targets['Validation Loss']:.2f}")
        ]

        for metric in metrics:
            item = self.stats_tree.insert('', 'end', values=metric)
            if metric[0] == 'Accuracy' and current_acc >= self.targets['Accuracy']:
                self.stats_tree.tag_configure('acc_achieved', foreground='green')
                self.stats_tree.item(item, tags=('acc_achieved',))
            elif metric[0] == 'Val Accuracy' and current_val_acc >= self.targets['Validation Accuracy']:
                self.stats_tree.tag_configure('val_acc_achieved', foreground='green')
                self.stats_tree.item(item, tags=('val_acc_achieved',))

    def update_plots(self):
        self.ax1.clear()
        self.ax2.clear()

        if self.generator and self.generator.training_history['accuracy']:
            epochs = range(1, len(self.generator.training_history['accuracy']) + 1)

            self.ax1.plot(epochs, self.generator.training_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            self.ax1.plot(epochs, self.generator.training_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Accuracy')
            self.ax1.set_title('Model Accuracy')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            self.ax1.axhline(y=self.targets['Accuracy'], color='g', linestyle='--', alpha=0.5, label='Target')

            self.ax2.plot(epochs, self.generator.training_history['loss'], 'b-', label='Training Loss', linewidth=2)
            self.ax2.plot(epochs, self.generator.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Loss')
            self.ax2.set_title('Model Loss')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            self.ax2.axhline(y=self.targets['Loss'], color='g', linestyle='--', alpha=0.5, label='Target')

        self.figure.tight_layout()
        self.canvas.draw()

    def start_training(self):
        if self.training:
            self.log_handler.log("Training already in progress", "WARNING")
            return

        if not self.generator:
            self.generator = TextGenerator(mode=self.mode_var.get())

        text = self.train_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter training text!")
            return

        char_count = len(text)
        word_count = len(text.split())

        self.log_handler.log(f"Training text stats: {char_count} characters, {word_count} words", "INFO")

        # Provide recommendations based on text size
        if self.mode_var.get() == 'character' and char_count < 1000:
            self.log_handler.log("Warning: Very small character dataset (<1000 chars). Results may be poor.", "WARNING")
        elif self.mode_var.get() == 'word' and word_count < 1000:
            self.log_handler.log("Warning: Small word dataset (<1000 words). Consider character mode or more text.", "WARNING")

        min_length = self.context_size.get() + 10
        if len(text) < min_length:
            messagebox.showerror("Error", f"Training text too short! Need at least {min_length} {'words' if self.mode_var.get() == 'word' else 'characters'}.")
            return

        self.training = True
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress['value'] = 0
        self.status_label.configure(text="Initializing training...")

        self.log_handler.log(f"Starting {self.mode_var.get()} level training with context size {self.context_size.get()}, {self.epochs.get()} epochs", "INFO")
        self.log_handler.log(f"Early stopping {'enabled' if self.use_early_stopping.get() else 'disabled'} (patience={self.patience.get()})", "INFO")

        # Run training in separate thread
        self.training_thread = Thread(target=self.training_worker, args=(text,), daemon=True)
        self.training_thread.start()

    def training_worker(self, text):
        """Worker function for training in separate thread"""
        try:
            self.generator.train(
                text=text,
                sequence_length=self.context_size.get(),
                epochs=self.epochs.get(),
                batch_size=self.batch_size.get(),
                validation_split=0.2,
                callback=lambda e, l, a, vl, va: self.root.after(0, self.update_training_status, e, l, a, vl, va),
                embedding_dim=self.embedding_dim.get(),
                lstm_units=self.lstm_units.get(),
                dropout_rate=self.dropout_rate.get(),
                use_early_stopping=self.use_early_stopping.get(),
                patience=self.patience.get()
            )
            self.root.after(0, self.training_complete, True, "Training completed successfully!")
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.root.after(0, self.training_complete, False, f"Training failed: {str(e)}")

    def stop_training(self):
        """Stop the training process"""
        if self.training and self.generator:
            self.log_handler.log("Stopping training...", "WARNING")
            self.generator.stop_training()
            self.status_label.configure(text="Stopping training...")

    def training_complete(self, success, message):
        self.training = False
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if success:
            self.log_handler.log(message, "SUCCESS")
            messagebox.showinfo("Success", message)
            self.status_label.configure(text="Training completed")

            stats = self.generator.get_model_stats()
            if stats:
                self.log_handler.log(f"Final Model Stats - Accuracy: {stats['final_accuracy']:.4f}, Loss: {stats['final_loss']:.4f}", "INFO")
                self.log_handler.log(f"Best Accuracy: {stats['best_accuracy']:.4f}, Best Validation Accuracy: {stats['best_val_accuracy']:.4f}", "INFO")
                self.log_handler.log(f"Epochs completed: {stats['epochs_completed']}/{self.epochs.get()}", "INFO")

                # Provide recommendations based on final stats
                if stats['best_val_accuracy'] < 0.5:
                    self.log_handler.log("Recommendation: Validation accuracy is low (<50%). Try:", "WARNING")
                    self.log_handler.log("  - Using smaller context size", "WARNING")
                    self.log_handler.log("  - More training text", "WARNING")
                    self.log_handler.log("  - More epochs or disable early stopping", "WARNING")
                elif stats['best_val_accuracy'] > 0.8:
                    self.log_handler.log("Excellent! Model is performing well.", "SUCCESS")
        else:
            self.log_handler.log(message, "ERROR")
            if "stopped by user" not in message.lower():
                messagebox.showerror("Error", message)
            self.status_label.configure(text="Training failed or stopped")

    def save_model(self):
        if not self.generator or not self.generator.model:
            messagebox.showerror("Error", "No model to save! Train or load a model first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )

        if filepath:
            base_path = filepath.replace("_model.h5", "").replace(".h5", "")
            if self.generator.save_model(base_path):
                self.log_handler.log(f"Model saved to {base_path}", "SUCCESS")
                messagebox.showinfo("Success", f"Model saved to {base_path}")
            else:
                self.log_handler.log("Failed to save model", "ERROR")
                messagebox.showerror("Error", "Failed to save model!")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )

        if filepath:
            base_path = filepath.replace("_model.h5", "").replace(".h5", "")
            temp_generator = TextGenerator(mode=self.mode_var.get())
            if temp_generator.load_model(base_path):
                self.generator = temp_generator
                self.mode_var.set(self.generator.mode)
                self.update_mode_info()
                self.log_handler.log(f"{self.generator.mode.upper()} level model loaded from {base_path}", "SUCCESS")
                messagebox.showinfo("Success", f"{self.generator.mode.upper()} level model loaded successfully!")
                self.status_label.configure(text=f"Model loaded - {self.generator.mode} mode")
                self.update_plots()

                stats = self.generator.get_model_stats()
                if stats:
                    self.log_handler.log(f"Loaded Model Stats - Completed Epochs: {stats['epochs_completed']}", "INFO")
                    self.log_handler.log(f"Final Accuracy: {stats['final_accuracy']:.4f}, Best Accuracy: {stats['best_accuracy']:.4f}", "INFO")
            else:
                self.log_handler.log("Failed to load model - missing files", "ERROR")
                messagebox.showerror("Error", "Failed to load model! Make sure tokenizer and config files exist.")

    def generate_text(self):
        if not self.generator or not self.generator.model:
            messagebox.showerror("Error", "No model loaded! Train or load a model first.")
            return

        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt!")
            return

        try:
            mode_name = "word" if self.generator.mode == 'word' else "character"
            self.log_handler.log(f"Generating {mode_name} text with prompt: '{prompt[:50]}...'", "INFO")
            start_time = time.time()

            generated = self.generator.generate_text(
                prompt=prompt,
                num_predictions=self.length_to_gen.get(),
                temperature=self.temperature.get()
            )

            generation_time = time.time() - start_time

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, generated)

            unit = "words" if self.generator.mode == 'word' else "characters"
            self.log_handler.log(f"Generated {self.length_to_gen.get()} {unit} in {generation_time:.2f} seconds", "SUCCESS")
        except Exception as e:
            self.log_handler.log(f"Generation failed: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Generation failed: {str(e)}")

def main():
    root = tk.Tk()
    app = TextGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
