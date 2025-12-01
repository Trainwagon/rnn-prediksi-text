import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, GRU, Bidirectional

# Konfigurasi Halaman
st.set_page_config(page_title="Aplikasi Prediksi Teks RNN", layout="wide")

st.title("🤖 Aplikasi Prediksi Teks dengan Deep Learning")
st.markdown("""
Aplikasi ini mendemonstrasikan penggunaan **Recurrent Neural Networks (RNN)** untuk memprediksi karakter selanjutnya dalam sebuah teks.
Anda dapat memilih varian model yang berbeda seperti yang dipelajari di kelas.
""")

# Sidebar untuk Parameter
st.sidebar.header("⚙️ Konfigurasi Model")

model_type = st.sidebar.selectbox(
    "Pilih Arsitektur Model",
    ["Vanilla RNN", "Bidirectional RNN", "LSTM", "GRU"],
    help="Pilih jenis layer RNN yang akan digunakan."
)

st.sidebar.subheader("Hyperparameters")
epochs = st.sidebar.number_input("Jumlah Epochs", min_value=10, value=100, step=10)
hidden_units = st.sidebar.number_input("Jumlah Neuron (Hidden Units)", min_value=10, value=50, step=10)
seq_length = st.sidebar.number_input("Panjang Urutan (Sequence Length)", min_value=3, value=5)
learning_rate = st.sidebar.selectbox("Learning Rate (Adam)", [0.001, 0.01, 0.0001], index=0)

# Layout Utama
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Data Latih (Training)")
    st.info("Masukkan teks yang akan dipelajari oleh model.")
    
    default_text = "Ini adalah contoh text yang diaktifkan oleh Recurrent Neural Networks"
    training_text = st.text_area("Input Teks Latih:", value=default_text, height=200)
    
    train_btn = st.button("🚀 Latih Model", type="primary")

    if train_btn:
        if len(training_text) <= seq_length:
            st.error(f"Teks terlalu pendek! Harap masukkan lebih dari {seq_length} karakter.")
        else:
            with st.spinner(f"Sedang melatih model {model_type}..."):
                # 1. Preprocessing
                chars = sorted(list(set(training_text)))
                char_to_index = {char: i for i, char in enumerate(chars)}
                index_to_char = {i: char for i, char in enumerate(chars)}
                num_chars = len(chars)
                
                sequences = []
                labels = []
                
                for i in range(len(training_text) - seq_length):
                    seq = training_text[i:i + seq_length]
                    label = training_text[i + seq_length]
                    sequences.append([char_to_index[char] for char in seq])
                    labels.append(char_to_index[label])
                
                X = np.array(sequences)
                y = np.array(labels)
                
                X_one_hot = tf.one_hot(X, num_chars)
                y_one_hot = tf.one_hot(y, num_chars)
                
                # 2. Membangun Model
                model = Sequential()
                input_shape = (seq_length, num_chars)
                
                if model_type == "Vanilla RNN":
                    # Sesuai notebook: SimpleRNN dengan relu
                    model.add(SimpleRNN(units=hidden_units, activation='relu', input_shape=input_shape))
                elif model_type == "Bidirectional RNN":
                    # Bidirectional wrapper pada SimpleRNN
                    model.add(Bidirectional(SimpleRNN(units=hidden_units, activation='relu'), input_shape=input_shape))
                elif model_type == "LSTM":
                    # LSTM standard biasanya menggunakan tanh
                    model.add(LSTM(units=hidden_units, activation='tanh', input_shape=input_shape))
                elif model_type == "GRU":
                    # GRU standard biasanya menggunakan tanh
                    model.add(GRU(units=hidden_units, activation='tanh', input_shape=input_shape))
                
                model.add(Dense(num_chars, activation='softmax'))
                
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                
                # 3. Training
                history = model.fit(X_one_hot, y_one_hot, epochs=epochs, verbose=0)
                
                # Simpan ke session state
                st.session_state['model'] = model
                st.session_state['char_to_index'] = char_to_index
                st.session_state['index_to_char'] = index_to_char
                st.session_state['chars'] = chars
                st.session_state['seq_length'] = seq_length
                st.session_state['model_type'] = model_type
                st.session_state['training_text'] = training_text # Simpan teks latih untuk referensi
                
                st.success("Model berhasil dilatih!")
                st.metric("Akurasi Akhir", f"{history.history['accuracy'][-1]:.2%}")
                
                # Plot loss
                st.line_chart(history.history['loss'])
                st.caption("Grafik Loss selama training")

with col2:
    st.header("2. Prediksi / Generasi Teks")
    st.info("Gunakan model yang sudah dilatih untuk memprediksi kelanjutan teks.")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Silakan latih model terlebih dahulu di panel sebelah kiri.")
    else:
        st.write(f"**Model Aktif:** {st.session_state['model_type']}")
        
        seed_text = st.text_input("Masukkan Kata Awal (Seed):", value="Ini a")
        num_generate = st.number_input("Jumlah Karakter Prediksi:", min_value=1, value=50)
        
        if st.button("✨ Mulai Prediksi"):
            model = st.session_state['model']
            char_to_index = st.session_state['char_to_index']
            index_to_char = st.session_state['index_to_char']
            seq_length_saved = st.session_state['seq_length']
            chars = st.session_state['chars']
            num_chars = len(chars)
            
            generated_text = seed_text
            current_seq = seed_text
            
            # Validasi karakter input
            invalid_chars = [c for c in seed_text if c not in char_to_index]
            if invalid_chars:
                st.error(f"Error: Karakter berikut tidak ada dalam data latih: {', '.join(invalid_chars)}")
                st.warning("Tips: Pastikan kata awal hanya menggunakan karakter yang ada di teks latih.")
            else:
                # Padding jika seed lebih pendek dari seq_length
                if len(current_seq) < seq_length_saved:
                    padding_char = " " if " " in char_to_index else chars[0]
                    padding = padding_char * (seq_length_saved - len(current_seq))
                    current_seq = padding + current_seq
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(num_generate):
                    # Ambil seq_length karakter terakhir
                    input_seq = current_seq[-seq_length_saved:]
                    
                    # Encoding
                    x_indices = [char_to_index[char] for char in input_seq]
                    x = np.array([x_indices])
                    x_one_hot = tf.one_hot(x, num_chars)
                    
                    # Prediksi
                    prediction = model.predict(x_one_hot, verbose=0)
                    next_index = np.argmax(prediction)
                    next_char = index_to_char[next_index]
                    
                    generated_text += next_char
                    current_seq += next_char
                    
                    progress_bar.progress((i + 1) / num_generate)
                
                status_text.text("Selesai!")
                
                st.subheader("Hasil Prediksi:")
                st.success(generated_text)
                
                st.markdown("---")
                st.caption("Catatan: Hasil prediksi bergantung pada kualitas dan kuantitas data latih serta jumlah epoch.")
