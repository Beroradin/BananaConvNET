import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# Configuração da Página
st.set_page_config(
    page_title="BananaConvNET",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS Customizados ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
    }
    
    /* Estilo para os Cards de Métricas (Diagnóstico e Confiança) */
    div[data-testid="stMetric"] {
        background-color: #ffffff; /* Fundo Branco */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }

    /* Forçar cor do texto para escuro dentro do card branco */
    div[data-testid="stMetric"] label {
        color: #555555 !important; /* Cor do título (ex: Doença Detectada) */
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Cor do valor (ex: Pestalotiopsis) */
    }
    
    /* Ajuste extra para mobile: garantir que as colunas não fiquem espremidas */
    @media (max-width: 640px) {
        div[data-testid="stMetric"] {
            margin-bottom: 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Configurações & Constantes ---
SIZE = 128
CLASSES = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']
NUM_CLASSES = len(CLASSES)

# --- 1. Definição da Arquitetura (API Funcional) ---
def create_improved_model(input_shape=(SIZE, SIZE, 3), num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.15)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.15)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.15)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name="last_conv_layer")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- 2. Carregamento do Modelo ---
@st.cache_resource
def load_app_model():
    try:
        model = create_improved_model()
        model.load_weights('banana_leaf_cnn.h5')
        print("Modelo carregado com sucesso.")
        return model
    except Exception as e:
        st.error(f"Erro crítico ao carregar modelo: {e}")
        return None

# --- 3. Funções Auxiliares ---
def normalize_single_image(img):
    img_normalized = np.clip((img - np.mean(img)) / (np.std(img) + 1e-7), -3, 3)
    return img_normalized

def preprocess_image(image_bytes):
    img = Image.open(image_bytes).convert('RGB')
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (SIZE, SIZE))
    img_normalized = normalize_single_image(img_resized)
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, img_resized

def get_grad_cam(model, img_array, predicted_class_index, last_conv_layer_name="last_conv_layer"):
    grad_model = Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-7)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    # AQUI ESTÁ A MÁGICA DA MISTURA COM O ALPHA AJUSTÁVEL
    superimposed_img = (heatmap_color_rgb * alpha + original_img * (1 - alpha)).astype(np.uint8)
    return superimposed_img

# --- 4. Interface do Usuário (Sidebar) ---

st.sidebar.title("Configurações")
st.sidebar.info("Esta ferramenta utiliza IA (Redes Neurais Convolucionais) para detectar doenças em folhas de bananeira.")

st.sidebar.markdown("---")
st.sidebar.subheader("Visualização")
# Slider para controlar a transparência do Grad-CAM
alpha_slider = st.sidebar.slider("Intensidade do Mapa de Calor", 0.0, 1.0, 0.6, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Desenvolvido com TensorFlow & Streamlit por Matheus Pereira Alves com base no artigo **")

# --- 5. Interface Principal ---

st.title("🍌 BananaConvNET")
st.markdown("### Diagnóstico Inteligente de Patologias Vegetais")
st.markdown("---")

model = load_app_model()

if model is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    # Coluna da Esquerda: Input
    with col1:
        st.subheader("Imagem da Folha")
        uploaded_file = st.file_uploader("Arraste ou selecione uma imagem", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            processed_batch, original_img_resized = preprocess_image(uploaded_file)
            st.image(uploaded_file, caption="Imagem Original", width="stretch")
        else:
            # Placeholder visual quando não tem imagem
            st.info("Aguardando upload para iniciar a análise...")

    # Coluna da Direita: Resultados
    with col2:
        if uploaded_file:
            st.subheader("Diagnóstico")
            
            with st.spinner('Processando redes neurais...'):
                # Predição
                predictions = model.predict(processed_batch)
                probs = predictions[0]
                predicted_idx = np.argmax(probs)
                predicted_label = CLASSES[predicted_idx]
                confidence = probs[predicted_idx]

                # Exibição de Destaque (Metric)
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    label_clean = predicted_label.capitalize()
                    st.metric(label="Doença Detectada", value=label_clean)
                
                with metric_col2:
                    st.metric(label="Nível de Confiança", value=f"{confidence:.1%}")

                # Barra de Progresso colorida
                if confidence > 0.8:
                    st.progress(float(confidence), text="Alta certeza no diagnóstico")
                elif confidence > 0.5:
                    st.warning("Certeza moderada. Verifique visualmente.")
                    st.progress(float(confidence))
                else:
                    st.error("Baixa certeza. Resultado inconclusivo.")
                    st.progress(float(confidence))

                # Detalhes em Expander
                with st.expander("Ver probabilidades detalhadas"):
                    df_probs = pd.DataFrame({'Doença': CLASSES, 'Probabilidade': probs})
                    st.bar_chart(df_probs.set_index('Doença'))

                # Grad-CAM Area
                st.divider()
                st.subheader("Análise de Foco (Grad-CAM)")
                st.write("Onde a IA encontrou o problema?")
                
                try:
                    heatmap = get_grad_cam(model, processed_batch, predicted_idx)
                    # Usa o valor do slider (alpha_slider) aqui
                    overlay = overlay_heatmap(original_img_resized, heatmap, alpha=alpha_slider)
                    
                    st.image(overlay, caption=f"Heatmap: {predicted_label}", width="stretch")
                except Exception as e:
                    st.error(f"Erro ao gerar visualização: {e}")
        
        else:
            # Texto placeholder à direita
            st.write("Faça o upload de uma imagem para ver o resultado aqui.")
