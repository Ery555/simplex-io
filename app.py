import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import base64
import os
from streamlit_option_menu import option_menu
from modules.linear_solver import resolver_lp
from modules.generador_reportes import generar_reporte_pdf
from modules.generador_reportes import generar_reporte_pdf_transporte
from modules.generador_reportes import generar_reporte_pdf_asignacion
from modules.generador_reportes import generar_reporte_pdf_redes


# Configuración de la página
st.set_page_config(page_title="SIMPLEX IO - UMSA", layout="wide")
st.markdown("""
    <style>
    /* Estilizar las métricas para que parezcan tarjetas modernas */
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 5% 10%;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    /* Ocultar el botón de menú de Streamlit por defecto para un look más "App" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
# Inyectar CDN de Bootstrap Icons para usar en toda la página
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

def titulo_con_icono(texto, icono, nivel="h2"):
    """
    Genera un encabezado HTML con un ícono de Bootstrap.
    Nivel h2 equivale a st.header, h3 equivale a st.subheader.
    """
    html = f"""
    <{nivel} style='color: #F4F4F5; font-weight: 600; margin-bottom: 1rem;'>
        <i class='bi bi-{icono}' style='color: #00D287; margin-right: 10px;'></i>{texto}
    </{nivel}>
    """
    st.markdown(html, unsafe_allow_html=True)

def obtener_base64_de_archivo(ruta_archivo):
    with open(ruta_archivo, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
# --- 1. INICIALIZACIÓN GLOBAL PROG LINEAL ---
if 'df_obj' not in st.session_state:
    st.session_state.df_obj = pd.DataFrame([[0.0, 0.0]], columns=['x1', 'x2'], index=['Coeficiente'])
if 'df_restr' not in st.session_state:
    st.session_state.df_restr = pd.DataFrame(
        {'x1': [0.0, 0.0], 'x2': [0.0, 0.0], 'Signo': ['<=', '<='], 'RHS': [0.0, 0.0]},
        index=['R1', 'R2']
    )
if 'tipo_opt' not in st.session_state:
    st.session_state.tipo_opt = "Maximizar"
# --- INICIALIZACIÓN GLOBAL TRANSPORTE ---
if 'df_transp' not in st.session_state:
    # Matriz inicial 2x2. Incluye Oferta y Demanda.
    # La esquina inferior derecha [Demanda, Oferta] se queda en 0 (no se usa)
    data_t = {
        'D1': [0.0, 0.0, 0.0],
        'D2': [0.0, 0.0, 0.0],
        'Oferta': [0.0, 0.0, 0.0] 
    }
    st.session_state.df_transp = pd.DataFrame(data_t, index=['O1', 'O2', 'Demanda'])

if 'tipo_opt_transp' not in st.session_state:
    st.session_state.tipo_opt_transp = "Minimizar"

# --- INICIALIZACIÓN: ASIGNACIÓN ---
if 'df_asign' not in st.session_state:
    # Matriz inicial limpia 3x3 (Trabajadores vs Tareas)
    st.session_state.df_asign = pd.DataFrame(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        index=['Trabajador 1', 'Trabajador 2', 'Trabajador 3'],
        columns=['Tarea 1', 'Tarea 2', 'Tarea 3']
    )
if 'tipo_opt_asign' not in st.session_state:
    st.session_state.tipo_opt_asign = "Minimizar"


# --- INICIALIZACIÓN: FLUJO DE REDES ---
if 'df_redes' not in st.session_state:
    # Columnas estándar para grafos: Desde, Hacia, Capacidad, Costo
    st.session_state.df_redes = pd.DataFrame(
        [["O", "A", 10.0, 2.0], ["O", "B", 15.0, 5.0], ["A", "T", 10.0, 1.0]],
        columns=['Nodo Origen', 'Nodo Destino', 'Capacidad', 'Costo Unitario']
    )
if 'tipo_problema_redes' not in st.session_state:
    st.session_state.tipo_problema_redes = "Flujo Máximo"

# FUNCIONES DE AJUSTE DE DIMENSIONES

# PROG LINEAL
def ajustar_dimensiones():
    """
    Sincroniza los cambios del diccionario del editor con el DataFrame 
    antes de redimensionar para evitar el AttributeError y el parpadeo.
    """
    # A. Sincronizar Función Objetivo
    if 'editor_obj' in st.session_state:
        # Extraemos solo las filas editadas del diccionario
        edits = st.session_state.editor_obj.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_obj.at[st.session_state.df_obj.index[row_idx], col] = val

    # B. Sincronizar Restricciones
    if 'editor_restr' in st.session_state:
        edits_r = st.session_state.editor_restr.get('edited_rows', {})
        for row_idx, changes in edits_r.items():
            for col, val in changes.items():
                st.session_state.df_restr.at[st.session_state.df_restr.index[row_idx], col] = val

    # C. Redimensionamiento Seguro
    n_v = st.session_state.n_vars_slider
    n_c = st.session_state.n_cons_slider
    new_vars = [f"x{i+1}" for i in range(n_v)]
    new_restr = [f"R{i+1}" for i in range(n_c)]

    # Reindexar usando el DataFrame real (ya actualizado con los cambios)
    st.session_state.df_obj = st.session_state.df_obj.reindex(columns=new_vars, fill_value=0.0)
    
    cols_restr = new_vars + ["Signo", "RHS"]
    st.session_state.df_restr = st.session_state.df_restr.reindex(columns=cols_restr, index=new_restr)
    
    # Rellenar valores por defecto en nuevas celdas
    st.session_state.df_restr[new_vars] = st.session_state.df_restr[new_vars].fillna(0.0)
    st.session_state.df_restr["Signo"] = st.session_state.df_restr["Signo"].fillna("<=")
    st.session_state.df_restr["RHS"] = st.session_state.df_restr["RHS"].fillna(0.0)

# --- TRANSPORTE ---
def ajustar_dimensiones_transp():
    """Redimensiona la matriz de Transporte protegiendo Oferta y Demanda"""
    # 1. Sincronizar datos manuales
    if 'editor_transp' in st.session_state:
        edits = st.session_state.editor_transp.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_transp.at[st.session_state.df_transp.index[row_idx], col] = val

    n_o = st.session_state.n_orig_slider
    n_d = st.session_state.n_dest_slider
    
    df = st.session_state.df_transp.copy()

    # 2. "Despegar" la fila de Demanda y columna de Oferta
    demanda_row = df.loc[['Demanda']].copy()
    df = df.drop(index='Demanda')
    
    oferta_col = df[['Oferta']].copy()
    df = df.drop(columns='Oferta')
    
    # 3. Redimensionar la matriz central de Costos
    new_origs = [f"O{i+1}" for i in range(n_o)]
    new_dests = [f"D{i+1}" for i in range(n_d)]
    df = df.reindex(index=new_origs, columns=new_dests, fill_value=0.0)
    
    # 4. Redimensionar los bordes
    demanda_row = demanda_row.drop(columns='Oferta').reindex(columns=new_dests, fill_value=0.0)
    oferta_col = oferta_col.reindex(index=new_origs, fill_value=0.0)
    
    # 5. Volver a "Pegar" todo
    df['Oferta'] = oferta_col['Oferta']
    demanda_row['Oferta'] = 0.0 # Esquina vacía
    st.session_state.df_transp = pd.concat([df, demanda_row])

# ASIGNACION

def ajustar_dimensiones_asign():
    """Redimensiona la matriz de Asignación guardando los datos previos."""
    # 1. Sincronizar edición manual
    if 'editor_asign' in st.session_state:
        edits = st.session_state.editor_asign.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_asign.at[st.session_state.df_asign.index[row_idx], col] = val

    # 2. Redimensionar
    n_w = st.session_state.n_workers_slider
    n_t = st.session_state.n_tasks_slider
    
    new_workers = [f"Trabajador {i+1}" for i in range(n_w)]
    new_tasks = [f"Tarea {i+1}" for i in range(n_t)]
    
    st.session_state.df_asign = st.session_state.df_asign.reindex(
        index=new_workers, columns=new_tasks, fill_value=0.0
    )

# FLUJO DE REDES

def ajustar_dimensiones_redes():
    """Redimensiona la tabla de arcos (filas) para el modelo de Redes."""
    if 'editor_redes' in st.session_state:
        edits = st.session_state.editor_redes.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_redes.at[st.session_state.df_redes.index[row_idx], col] = val

    n_arcos = st.session_state.n_arcos_slider
    
    # Solo ajustamos el número de filas (añadimos vacías o recortamos)
    current_len = len(st.session_state.df_redes)
    if n_arcos > current_len:
        # Añadir filas nuevas
        nuevas_filas = pd.DataFrame(
            [["", "", 0.0, 0.0] for _ in range(n_arcos - current_len)],
            columns=st.session_state.df_redes.columns
        )
        st.session_state.df_redes = pd.concat([st.session_state.df_redes, nuevas_filas], ignore_index=True)
    elif n_arcos < current_len:
        # Recortar filas sobrantes
        st.session_state.df_redes = st.session_state.df_redes.iloc[:n_arcos]


# Menú lateral
with st.sidebar:
    # 1. Cargamos el logo local (Asegúrate de poner la ruta correcta de tu archivo)
    try:
        # Reemplaza 'ruta/a/tu/logo_umsa.png' por el nombre real de tu archivo
         
        ruta_logo = os.path.join("assets", "Logo_Umsa.png") # Ajusta 'assets' si tu carpeta tiene otro nombre
        if os.path.exists(ruta_logo):
            logo_base64 = obtener_base64_de_archivo(ruta_logo)
            # ... resto de tu lógica de visualización ...
        else:
            st.sidebar.error("No se encontró el archivo del logo en la ruta especificada.")
        
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{logo_base64}" width="100" style="margin-bottom: 10px;">
                <h2 style='color: #00D287; margin-bottom: 0;'>UMSA - FCPN</h2>
                <p style='color: #888; font-size: 0.9rem;'>Investigación Operativa I</p>
            </div>
            <hr style="margin-top: 5px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
    except Exception:
        # Si la imagen no carga, muestra solo el texto para que no de error
        st.markdown("<h2 style='text-align: center; color: #00D287;'>UMSA - FCPN</h2>", unsafe_allow_html=True)
    
    # Menú avanzado con íconos de FontAwesome
    opcion = option_menu(
        menu_title="Módulos",
        options=["Inicio", "Programación Lineal", "Modelo de Transporte", "Asignación", "Flujo de Redes"],
        icons=["house-fill", "graph-up-arrow", "truck", "people-fill", "diagram-3-fill"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00D287", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin":"5px", 
                "--hover-color": "#27272A"
            },
            "nav-link-selected": {"background-color": "#27272A", "color": "white"},
        }
    )
    
    st.markdown("---")
    st.caption("⚙️ Motor: PuLP + CBC")

# Lógica de navegación
# --- INTERFAZ DE LANDING PAGE -----------------------------------------------------------------------------
if opcion == "Inicio":
    # --- ESTILOS CSS ADICIONALES PARA LA LANDING PAGE ---
    st.markdown("""
        <style>
        .hero-section {
            padding: 60px 20px;
            text-align: center;
            background: linear-gradient(135deg, #09090B 0%, #18181B 100%);
            border-radius: 20px;
            border: 1px solid #27272A;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 40px;
        }
        .feature-card {
            background: #18181B;
            padding: 30px;
            border-radius: 15px;
            border: 1px solid #27272A;
            transition: all 0.3s ease;
            text-align: center;
            height: 100%;
        }
        .feature-card:hover {
            border-color: #00D287;
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 210, 135, 0.2);
        }
        .icon-circle {
            width: 60px;
            height: 60px;
            background: rgba(0, 210, 135, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            color: #00D287;
            font-size: 24px;
        }
        .highlight-text {
            color: #00D287;
            font-weight: bold;
        }
        /* Animaciones estilo Framer Motion / Tailwind */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-hero {
            animation: fadeInUp 0.8s ease-out forwards;
        }
        
        /* Botón CTA estilo Tailwind */
        .cta-button {
            display: inline-block;
            background-color: #00D287;
            color: #09090B;
            padding: 12px 28px;
            border-radius: 8px;
            font-weight: bold;
            text-decoration: none;
            margin-top: 25px;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
        }
        .cta-button:hover {
            background-color: #00FFAA;
            box-shadow: 0 0 20px rgba(0, 210, 135, 0.4);
            transform: scale(1.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # --- SECCIÓN HERO ---
    st.markdown(f"""
        <div class="hero-section animate-hero">
            <h1 style='font-size: 3.5rem; color: #F4F4F5; margin-bottom: 10px;'>
                SIMPLEX <span style='color: #00D287;'>IO</span>
            </h1>
            <p style='font-size: 1.2rem; color: #A1A1AA; max-width: 800px; margin: 0 auto;'>
                La plataforma definitiva para la optimización matemática y la toma de decisiones estratégicas. Transformando la complejidad operativa en eficiencia absoluta.
            </p>
            <div style="margin-top: 25px;">
                <p style="color: #F4F4F5; font-size: 0.95rem; opacity: 0.8;">👈 Selecciona un módulo en el panel lateral para empezar</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- TEXTO INTRODUCTORIO ---
    col_intro1, col_intro2 = st.columns([2, 1])
    with col_intro1:
        titulo_con_icono("Sobre la Plataforma", "info-circle", "h2")
        st.write("""
            En la intersección entre la ciencia de datos, la economía y la ingeniería de software se encuentra la capacidad de tomar decisiones precisas. **SIMPLEX IO** nace como una solución arquitectónica diseñada para abordar problemas críticos de asignación de recursos, logística y planificación estratégica.
        """)
        st.write("""
            Desarrollada para el ecosistema académico y profesional, esta herramienta elimina la barrera entre la teoría matemática abstracta y su aplicación práctica. Al aprovechar motores de resolución de vanguardia como <span class='highlight-text'>PuLP y algoritmos de teoría de grafos</span>, permite a investigadores, analistas y líderes maximizar utilidades y minimizar costos operativos con un rigor cuantitativo absoluto.
        """, unsafe_allow_html=True)

    with col_intro2:
        st.markdown("""
            <div style="background: #18181B; padding: 20px; border-radius: 15px; border-left: 5px solid #00D287;">
                <h4 style="margin-top:0; color: #F4F4F5;">🚀 Flujo de Trabajo</h4>
                <p style="font-size: 0.9rem; color: #A1A1AA; line-height: 1.6;">
                    <strong>1. Modelado:</strong> Selecciona el algoritmo en el menú lateral.<br>
                    <strong>2. Parametrización:</strong> Define la dimensionalidad del sistema.<br>
                    <strong>3. Ingesta de Datos:</strong> Configura la matriz de costos o arcos.<br>
                    <strong>4. Ejecución:</strong> Computa el estado óptimo y exporta el análisis.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    titulo_con_icono("Módulos de Optimización Cuantitativa", "grid-fill", "h2")

    # --- TARJETAS DE MÓDULOS (Features) ---
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
            <div class="feature-card">
                <div class="icon-circle"><i class="bi bi-graph-up-arrow"></i></div>
                <h4 style="color: #F4F4F5;">Prog. Lineal</h4>
                <p style="font-size: 0.85rem; color: #A1A1AA;">
                    Resolución algorítmica multidimensional con análisis de sensibilidad profundo, evaluación de holguras y precios sombra para escenarios de maximización económica.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="feature-card">
                <div class="icon-circle"><i class="bi bi-truck"></i></div>
                <h4 style="color: #F4F4F5;">Transporte</h4>
                <p style="font-size: 0.85rem; color: #A1A1AA;">
                    Optimización avanzada de cadenas de suministro. Incorpora balanceo matricial automático y cálculo de multiplicadores duales para la auditoría de costos logísticos.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="feature-card">
                <div class="icon-circle"><i class="bi bi-people-fill"></i></div>
                <h4 style="color: #F4F4F5;">Asignación</h4>
                <p style="font-size: 0.85rem; color: #A1A1AA;">
                    Vinculación inteligente de agentes a tareas mediante programación entera binaria, garantizando la máxima rentabilidad en la distribución de capital humano.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
            <div class="feature-card">
                <div class="icon-circle"><i class="bi bi-diagram-3-fill"></i></div>
                <h4 style="color: #F4F4F5;">Redes</h4>
                <p style="font-size: 0.85rem; color: #A1A1AA;">
                    Modelado topológico de grafos dirigidos. Capacidad analítica para identificar cuellos de botella sistémicos y determinar rutas de latencia o distancia mínima.
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    titulo_con_icono("Aplicaciones y Casos de Uso Estratégicos", "lightbulb-fill", "h2")

    # Estilos específicos para los casos de uso
    st.markdown("""
        <style>
        .use-case-box {
            background: #18181B;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #00D287;
            margin-bottom: 20px;
            height: 100%;
        }
        .use-case-title {
            color: #00D287;
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .use-case-text {
            color: #A1A1AA;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)

    # Fila 1 de Casos de Uso
    cu_col1, cu_col2 = st.columns(2)

    with cu_col1:
        st.markdown("""
            <div class="use-case-box">
                <div class="use-case-title">
                    <i class="bi bi-briefcase-fill" style="margin-right:10px;"></i>
                    Optimización de Mix de Producción
                </div>
                <p class="use-case-text">
                    <strong>Escenario:</strong> Una fábrica dispone de recursos limitados (materia prima, horas hombre) para elaborar distintos productos con diferentes márgenes de utilidad.<br>
                    <strong>Aplicación:</strong> El modelo de <em>Programación Lineal</em> determina la cantidad exacta de cada producto a fabricar para maximizar el beneficio neto total respetando todas las restricciones operativas.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with cu_col2:
        st.markdown("""
            <div class="use-case-box">
                <div class="use-case-title">
                    <i class="bi bi-geo-alt-fill" style="margin-right:10px;"></i>
                    Logística y Distribución Nacional
                </div>
                <p class="use-case-text">
                    <strong>Escenario:</strong> Una cadena de suministros debe mover mercancía desde múltiples centros de acopio hacia diversos puntos de venta minorista en Bolivia.<br>
                    <strong>Aplicación:</strong> El <em>Modelo de Transporte</em> minimiza los costos de flete y logística, asegurando que se satisfaga la demanda de cada cliente sin exceder la capacidad de las plantas de origen.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Fila 2 de Casos de Uso
    cu_col3, cu_col4 = st.columns(2)

    with cu_col3:
        st.markdown("""
            <div class="use-case-box">
                <div class="use-case-title">
                    <i class="bi bi-person-gear" style="margin-right:10px;"></i>
                    Gestión de Talento y Proyectos
                </div>
                <p class="use-case-text">
                    <strong>Escenario:</strong> Una consultora de software necesita asignar líderes técnicos a proyectos específicos según su especialidad y costo por hora.<br>
                    <strong>Aplicación:</strong> El <em>Modelo de Asignación</em> resuelve el emparejamiento 1 a 1 de manera óptima, garantizando que el talento humano sea aprovechado de forma eficiente para reducir el gasto presupuestario.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with cu_col4:
        st.markdown("""
            <div class="use-case-box">
                <div class="use-case-title">
                    <i class="bi bi-cpu-fill" style="margin-right:10px;"></i>
                    Infraestructura de Datos y Redes
                </div>
                <p class="use-case-text">
                    <strong>Escenario:</strong> Una operadora de telecomunicaciones necesita determinar el ancho de banda máximo que puede fluir a través de su infraestructura sin saturar los nodos.<br>
                    <strong>Aplicación:</strong> Los algoritmos de <em>Flujo de Redes</em> identifican los cuellos de botella y las rutas críticas, optimizando la latencia y la capacidad de transmisión en sistemas distribuidos.
                </p>
            </div>
        """, unsafe_allow_html=True)
    

# --- 3. INTERFAZ DE PROGRAMACIÓN LINEAL -------------------------------------------------------------------
elif opcion == "Programación Lineal":
    st.title("🚀 Suite de Investigación Operativa")
    st.markdown("---")
    titulo_con_icono("Programación Lineal General", "graph-up-arrow", "h2")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Variables", 1, 10, len(st.session_state.df_obj.columns), 
                    key="n_vars_slider", on_change=ajustar_dimensiones)
        with col2:
            st.slider("Restricciones", 1, 15, len(st.session_state.df_restr), 
                    key="n_cons_slider", on_change=ajustar_dimensiones)
        with col3:
            st.radio("Objetivo", ["Maximizar", "Minimizar"], horizontal=True, key="tipo_opt")

    titulo_con_icono("Coeficientes de la Función Objetivo", "pencil-square", "h3")
    # Mostramos la tabla base. Streamlit se encarga de mostrar los cambios encima.
    st.data_editor(st.session_state.df_obj, key="editor_obj", use_container_width=True)

    titulo_con_icono("Restricciones del Modelo", "list-check", "h3")
    st.data_editor(st.session_state.df_restr, key="editor_restr", use_container_width=True, 
                column_config={
                    "Signo": st.column_config.SelectboxColumn(
                        options=["<=", ">=", "="],
                        required=True,
                    )
                })

    # --- BOTÓN DE RESOLUCIÓN COMPLETO ---
    st.markdown("---")
    if st.button("Resolver con PuLP", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # 1. FUSIÓN DE DATOS (Asegura que el solver lea lo que ves en pantalla)
        # Fusión Función Objetivo
        obj_final = st.session_state.df_obj.copy()
        if 'editor_obj' in st.session_state:
            for r_idx, changes in st.session_state.editor_obj.get('edited_rows', {}).items():
                for c, v in changes.items():
                    obj_final.at[obj_final.index[r_idx], c] = v

        # Fusión Restricciones
        restr_final = st.session_state.df_restr.copy()
        if 'editor_restr' in st.session_state:
            for r_idx, changes in st.session_state.editor_restr.get('edited_rows', {}).items():
                for c, v in changes.items():
                    restr_final.at[restr_final.index[r_idx], c] = v

        # 2. LLAMADA AL SOLVER
        
        resultado = resolver_lp(obj_final, restr_final, st.session_state.tipo_opt)

        # 3. DESPLIEGUE DE RESULTADOS
        if "error" in resultado:
            st.error(f"Error técnico: {resultado['error']}")
        elif resultado["status"] != "Optimal":
            st.warning(f"Atención: El problema no es Óptimo. Estado: {resultado['status']}")
        else:
            # Feedback inmediato de éxito
            st.toast("¡Optimización Exitosa!", icon="✅")
            st.balloons()
            
            # --- CREACIÓN DE PESTAÑAS (TABS) MODERNAS ---
            tab_res, tab_graf, tab_exp = st.tabs(["📊 Resultados Numéricos", "📈 Análisis Gráfico", "📄 Exportar PDF"])
            
            # --- PESTAÑA 1: RESULTADOS ---
            with tab_res:
                st.subheader("Métricas Principales")
                c1, c2 = st.columns(2)
                c1.metric("Estado de la Solución", resultado["status"])
                val_z = resultado.get("z", 0.0)
                c2.metric("Valor Óptimo Z", f"{val_z:,.2f}")

                titulo_con_icono("Valores Óptimos de las Variables", "check-circle-fill", "h3")
                df_vars = pd.DataFrame([resultado["variables"]])
                df_vars.index = [1] 
                st.dataframe(df_vars, use_container_width=True)

                titulo_con_icono("Análisis de Sensibilidad", "search", "h3")
                with st.expander("Ver Precios Sombra y Holguras", expanded=True):
                    df_sens = resultado["sensibilidad"].copy()
                    df_sens.index = df_sens.index + 1
                    st.dataframe(df_sens, use_container_width=True)

            # --- PESTAÑA 2: GRÁFICO 2D ---
            with tab_graf:
                if st.session_state.n_vars_slider == 2:
                    titulo_con_icono("Gráfico del Modelo Geometrico", "bar-chart-line", "h3")
                    
                    cols = list(obj_final.columns)
                    x1_name, x2_name = cols[0], cols[1]
                    
                    opt_x1 = resultado["variables"][x1_name]
                    opt_x2 = resultado["variables"][x2_name]
                    
                    max_val = max(10.0, opt_x1 * 1.5, opt_x2 * 1.5)
                    x_vals = np.linspace(0, max_val, 400)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.style.use("dark_background")
                    
                    for idx, row in restr_final.iterrows():
                        c1 = row[x1_name]
                        c2 = row[x2_name]
                        rhs = row['RHS']
                        signo = row['Signo']
                        
                        if c2 != 0:
                            y_vals = (rhs - c1 * x_vals) / c2
                            ax.plot(x_vals, y_vals, label=f"{idx} ({signo} {rhs})", linewidth=2)
                        else:
                            if c1 != 0:
                                x_vert = rhs / c1
                                ax.axvline(x=x_vert, label=f"{idx} ({signo} {rhs})", linewidth=2, color=np.random.rand(3,))

                    ax.plot(opt_x1, opt_x2, marker='*', color='red', markersize=20, label=f"Óptimo ({opt_x1:.2f}, {opt_x2:.2f})")
                    
                    ax.set_xlim(0, max_val)
                    ax.set_ylim(0, max_val)
                    ax.axhline(0, color='white', linewidth=1)
                    ax.axvline(0, color='white', linewidth=1)
                    ax.set_xlabel(f"Variable {x1_name}", fontsize=12, fontweight='bold')
                    ax.set_ylabel(f"Variable {x2_name}", fontsize=12, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    st.pyplot(fig)
                else:
                    # Mensaje elegante si no hay 2 variables
                    st.info("💡 El gráfico geométrico solo está disponible para modelos que tengan exactamente 2 variables de decisión.")

            # --- PESTAÑA 3: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Descarga el reporte detallado con formato institucional para adjuntarlo a tus trabajos.")
                
                ruta_pdf = generar_reporte_pdf(
                    "Programación Lineal", 
                    st.session_state.tipo_opt, 
                    resultado["z"], 
                    resultado["variables"], 
                    resultado.get("sensibilidad")
                )

                with open(ruta_pdf, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte en PDF",
                        data=f,
                        file_name=f"Reporte_Optimización_{st.session_state.tipo_opt}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

# --- 3. INTERFAZ DE MODELO DE TRANSPORTE -------------------------------------------------------------------


elif opcion == "Modelo de Transporte":
    st.title("🚀 Suite de Investigación Operativa")
    st.markdown("---")
    titulo_con_icono("Modelo de Transporte", "truck")
    
    # Configuración
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Determinamos cuántos Orígenes hay (restando la fila 'Demanda')
            current_origs = len(st.session_state.df_transp) - 1
            st.slider("Nº de Orígenes (Plantas)", 1, 15, current_origs, 
                      key="n_orig_slider", on_change=ajustar_dimensiones_transp)
        with col2:
            # Determinamos cuántos Destinos hay (restando la columna 'Oferta')
            current_dests = len(st.session_state.df_transp.columns) - 1
            st.slider("Nº de Destinos (Clientes)", 1, 15, current_dests, 
                      key="n_dest_slider", on_change=ajustar_dimensiones_transp)
        with col3:
            st.radio("Objetivo", ["Minimizar", "Maximizar"], horizontal=True, key="tipo_opt_transp")
            
    st.info("💡 **Instrucciones:** Ingresa los costos unitarios de envío en las celdas centrales. Coloca la disponibilidad en la columna **Oferta** y los requerimientos en la fila **Demanda**.")

    # Matriz Interactiva
    titulo_con_icono("Matriz de Costos, Oferta y Demanda", "grid-3x3", "h3")
    st.data_editor(st.session_state.df_transp, key="editor_transp", use_container_width=True)

# --- BOTÓN DE RESOLUCIÓN (Reemplazar en la sección Transporte) ---
    st.markdown("---")
    if st.button("Resolver Modelo de Transporte", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # 1. Consolidación de datos
        df_final_t = st.session_state.df_transp.copy()
        if 'editor_transp' in st.session_state:
            for r_idx, changes in st.session_state.editor_transp.get('edited_rows', {}).items():
                for c, v in changes.items():
                    df_final_t.at[df_final_t.index[r_idx], c] = v
        
        # 2. Llamar al Solver
        from modules.transport_solver import resolver_transporte
        resultado = resolver_transporte(df_final_t, st.session_state.tipo_opt_transp)

        # 3. Mostrar Resultados Modernos
        # 3. Mostrar Resultados Modernos
        if "error" in resultado:
            st.error(f"Error técnico: {resultado['error']}")
        elif resultado["status"] != "Optimal":
            st.warning(f"El problema no tiene solución óptima. Estado: {resultado['status']}")
        else:
            # Feedback interactivo
            st.toast("¡Plan de Transporte Óptimo Encontrado!", icon="✅")
            st.balloons()

            # --- CREACIÓN DE PESTAÑAS (TABS) ---
            tab_res, tab_duales, tab_exp = st.tabs(["📊 Matriz de Envíos", "🔄 Análisis Dual", "📄 Exportar PDF"])

            # --- PESTAÑA 1: RESULTADOS Y MATRIZ ---
            with tab_res:
                # Avisos de IO1: Balanceo
                if not resultado["balanceado"]:
                    st.info("ℹ️ **Nota Académica:** El modelo original estaba **Desbalanceado**. El programa creó un nodo ficticio automáticamente para absorber la diferencia sin afectar el costo.")
                
                # Métricas
                c1, c2 = st.columns(2)
                c1.metric("Estado", resultado["status"])
                val_z = resultado.get("z", 0.0)
                c2.metric("Costo/Beneficio Óptimo (Z)", f"{val_z:,.2f}")

                # Matriz de Resultados
                titulo_con_icono("Matriz Óptima de Envíos", "box-seam", "h3")
                st.write("Las celdas muestran la cantidad que debes enviar desde cada Origen a cada Destino.")
                st.dataframe(resultado["matriz_resultados"], use_container_width=True)

            # --- PESTAÑA 2: DUALES ---
            with tab_duales:
                titulo_con_icono("Multiplicadores Duales", "arrow-repeat", "h3")
                st.write("Valores duales ($u_i, v_j$) asociados a las restricciones de oferta y demanda del modelo:")
                
                with st.container(border=True):
                    c_dual_o, c_dual_d = st.columns(2)
                    with c_dual_o:
                        st.markdown("**Orígenes ($u_i$)**")
                        st.json(resultado["duales_oferta"])
                    with c_dual_d:
                        st.markdown("**Destinos ($v_j$)**")
                        st.json(resultado["duales_demanda"])

            # --- PESTAÑA 3: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Descarga el reporte detallado con la matriz de envíos y análisis de balanceo.")
                
                # Preparamos el PDF de Transporte
                ruta_pdf_t = generar_reporte_pdf_transporte(
                    st.session_state.tipo_opt_transp,
                    resultado["z"],
                    resultado["matriz_resultados"],
                    resultado["duales_oferta"],
                    resultado["duales_demanda"],
                    resultado["balanceado"]
                )

                with open(ruta_pdf_t, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte de Transporte (PDF)",
                        data=f,
                        file_name=f"Reporte_Transporte_{st.session_state.tipo_opt_transp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )




# --- MÓDULO: MODELO DE ASIGNACIÓN -----------------------------------------------------------------
elif opcion == "Asignación":
    st.title("🚀 Suite de Investigación Operativa")
    st.markdown("---")
    titulo_con_icono("Modelo de Asignación", "people-fill")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Nº de Trabajadores / Agentes", 1, 15, len(st.session_state.df_asign.index), 
                      key="n_workers_slider", on_change=ajustar_dimensiones_asign)
        with col2:
            st.slider("Nº de Tareas / Destinos", 1, 15, len(st.session_state.df_asign.columns), 
                      key="n_tasks_slider", on_change=ajustar_dimensiones_asign)
        with col3:
            st.radio("Objetivo", ["Minimizar", "Maximizar"], horizontal=True, key="tipo_opt_asign")
            
    st.info("💡 **Instrucciones:** Ingresa los costos, tiempos o utilidades en la matriz. El programa balanceará automáticamente si el número de trabajadores no coincide con el de tareas.")

    titulo_con_icono("Matriz de Asignación", "table", "h3")
    st.data_editor(st.session_state.df_asign, key="editor_asign", use_container_width=True)

    st.markdown("---")
    if st.button("Resolver Asignación", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # Consolidar datos
        df_final_a = st.session_state.df_asign.copy()
        if 'editor_asign' in st.session_state:
            for r_idx, changes in st.session_state.editor_asign.get('edited_rows', {}).items():
                for c, v in changes.items():
                    df_final_a.at[df_final_a.index[r_idx], c] = v
        
        # Llamar al Solver
        from modules.assignment_solver import resolver_asignacion
        resultado = resolver_asignacion(df_final_a, st.session_state.tipo_opt_asign)

        # Mostrar Resultados
        # Mostrar Resultados
        if "error" in resultado:
            st.error(f"Error técnico: {resultado['error']}")
        elif resultado["status"] != "Optimal":
            st.warning(f"No se encontró solución óptima. Estado: {resultado['status']}")
        else:
            # Feedback interactivo
            st.toast("¡Asignación Óptima Completada!", icon="✅")
            st.balloons()

            # --- CREACIÓN DE PESTAÑAS (TABS) ---
            tab_plan, tab_exp = st.tabs(["🎯 Plan de Asignación", "📄 Exportar PDF"])

            # --- PESTAÑA 1: PLAN Y MATRIZ ---
            with tab_plan:
                if not resultado["balanceado"]:
                    st.info(f"ℹ️ **Nota de Balanceo:** Había {resultado['n_trabajadores']} trabajadores y {resultado['n_tareas']} tareas. Se añadieron elementos ficticios automáticamente.")

                c1, c2 = st.columns(2)
                c1.metric("Estado", resultado["status"])
                val_z = resultado.get("z", 0.0)
                c2.metric("Valor Óptimo (Z)", f"{val_z:,.2f}")

                titulo_con_icono("Matriz de Asignación", "person-badge", "h3")
                st.write("A continuación se muestra la asignación óptima de tareas:")
                st.table(resultado["lista_asignaciones"])
                
                with st.expander("Ver Matriz Binaria de Decisión ($x_{ij}$)", expanded=False):
                    st.write("Representación matemática de la asignación (1 = Asignado, 0 = No asignado):")
                    st.dataframe(resultado["matriz_resultados"], use_container_width=True)

            # --- PESTAÑA 2: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Obtén un reporte formal en PDF con el plan de asignación y la matriz técnica.")
                
                # Preparamos el PDF de Asignación
                ruta_pdf_a = generar_reporte_pdf_asignacion(
                    st.session_state.tipo_opt_asign,
                    resultado["z"],
                    resultado["lista_asignaciones"],
                    resultado["matriz_resultados"],
                    resultado["balanceado"]
                )

                with open(ruta_pdf_a, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte de Asignación (PDF)",
                        data=f,
                        file_name=f"Reporte_Asignacion_{st.session_state.tipo_opt_asign}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )












# --- MÓDULO: FLUJO DE REDES ---
elif opcion == "Flujo de Redes":
    st.title("🚀 Suite de Investigación Operativa")
    st.markdown("---")
    titulo_con_icono("Flujo de Redes", "diagram-3-fill")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.slider("Número de Arcos (Rutas)", 1, 30, len(st.session_state.df_redes), 
                      key="n_arcos_slider", on_change=ajustar_dimensiones_redes)
        with col2:
            st.radio(
                "Tipo de Problema", 
                ["Flujo Máximo", "Ruta Más Corta"], 
                horizontal=True, 
                key="tipo_problema_redes"
            )
            
    st.info("💡 **Instrucciones:** Define tu grafo. Usa letras o números para los nodos (ej. 'O' para Origen, 'T' para Destino).")

    # Inputs de Nodos Inicio y Fin
    c_origen, c_destino = st.columns(2)
    with c_origen:
        st.text_input("📍 Nodo de Inicio (Source)", value="O", key="nodo_inicio")
    with c_destino:
        st.text_input("🏁 Nodo Final (Sink)", value="T", key="nodo_final")

    titulo_con_icono("Definición de Arcos", "vector-pen", "h3")
    
    tipo_prob = st.session_state.tipo_problema_redes
    column_config = {
        "Nodo Origen": st.column_config.TextColumn("Nodo Origen", required=True),
        "Nodo Destino": st.column_config.TextColumn("Nodo Destino", required=True),
    }
    
    # Ocultar columnas irrelevantes
    if tipo_prob == "Flujo Máximo":
        column_config["Costo Unitario"] = None
    elif tipo_prob == "Ruta Más Corta":
        column_config["Capacidad"] = None
        
    st.data_editor(
        st.session_state.df_redes, 
        key="editor_redes", 
        use_container_width=True,
        column_config=column_config,
        hide_index=True
    )

    st.markdown("---")
    if st.button("Resolver Red de Nodos", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # 1. Consolidación de datos
        df_final_r = st.session_state.df_redes.copy()
        if 'editor_redes' in st.session_state:
            for r_idx, changes in st.session_state.editor_redes.get('edited_rows', {}).items():
                for c, v in changes.items():
                    df_final_r.at[df_final_r.index[r_idx], c] = v

        from modules.network_solver import resolver_flujo_maximo, resolver_ruta_mas_corta
        origen = st.session_state.nodo_inicio.strip()
        destino = st.session_state.nodo_final.strip()

        # 2. Enrutador
        if tipo_prob == "Flujo Máximo":
            resultado = resolver_flujo_maximo(df_final_r, origen, destino)
            label_z = "Flujo Máximo Total"
        else: # Ruta Más Corta
            resultado = resolver_ruta_mas_corta(df_final_r, origen, destino)
            label_z = "Distancia Mínima Total"

        # 3. Mostrar Resultados
        if "error" in resultado:
            st.error(resultado["error"])
        elif resultado["status"] != "Optimal":
            st.warning("No se encontró una solución óptima para esta red.")
        else:
            # Feedback interactivo
            st.toast(f"¡Cálculo de {tipo_prob} Exitoso!", icon="✅")
            st.balloons()
            
            # --- CREACIÓN DE PESTAÑAS (TABS) ---
            tab_res, tab_graf, tab_exp = st.tabs(["🛤️ Resultados Numéricos", "🌐 Grafo Visual", "📄 Exportar PDF"])

            # --- PESTAÑA 1: RESULTADOS ---
            with tab_res:
                st.divider()
                c1, c2 = st.columns(2)
                c1.metric("Estado", "Óptimo")
                c2.metric(label_z, f"{resultado['valor']:,.2f}")
                
                titulo_con_icono("Detalle de Rutas Utilizadas", "geo-alt", "h3")
                if not resultado["rutas"].empty:
                    st.dataframe(resultado["rutas"], use_container_width=True)
                    
                    # Análisis de cuellos de botella exclusivo para Flujo Máximo
                    if tipo_prob == "Flujo Máximo":
                        df_rutas = resultado["rutas"]
                        rutas_saturadas = len(df_rutas[df_rutas["Capacidad Sobrante"] == 0])
                        st.info(f"💡 **Análisis de Cuellos de Botella:** Tienes **{rutas_saturadas}** rutas operando a su máxima capacidad (Sobrante = 0). Para mejorar el flujo total, debes expandir la capacidad de estas rutas.")
                else:
                    st.warning("No es posible establecer una ruta con esta configuración.")

            # --- PESTAÑA 2: VISUALIZACIÓN GRÁFICA ---
            with tab_graf:
                titulo_con_icono("Visualización del Grafo Óptimo", "share", "h3")
                st.write("Las rutas resaltadas en verde representan el flujo óptimo o la ruta más corta encontrada.")
                
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.style.use("dark_background") # Estilo oscuro para proteger la vista
                    
                    G = nx.DiGraph() 
                    
                    rutas_activas = []
                    for _, row in resultado["rutas"].iterrows():
                        rutas_activas.append((str(row["De"]), str(row["A"])))
                    
                    for _, row in df_final_r.iterrows():
                        u = str(row['Nodo Origen']).strip()
                        v = str(row['Nodo Destino']).strip()
                        if u and v: 
                            G.add_edge(u, v)
                    
                    pos = nx.spring_layout(G, seed=42) 
                    
                    nx.draw_networkx_nodes(G, pos, node_color='#9b59b6', node_size=700, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
                    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='#555555', arrows=True, ax=ax)
                    nx.draw_networkx_edges(G, pos, edgelist=rutas_activas, edge_color='#2ecc71', width=3.0, arrows=True, arrowsize=20, ax=ax)
                    
                    ax.set_title(f"Grafo de Solución: {tipo_prob}", color='white', pad=20)
                    plt.axis('off') 
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"No se pudo generar la visualización gráfica del grafo: {e}")

            # --- PESTAÑA 3: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Descarga un reporte formal en PDF con las rutas utilizadas y métricas generales del modelo de redes.")
                
                ruta_pdf_r = generar_reporte_pdf_redes(
                    tipo_prob,
                    origen,
                    destino,
                    resultado["valor"],
                    resultado["rutas"]
                )

                with open(ruta_pdf_r, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte de Redes (PDF)",
                        data=f,
                        file_name=f"Reporte_{tipo_prob.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
# --- FOOTER INSTITUCIONAL (Añadir al final del archivo app.py) ---
st.markdown(" <br><br> ", unsafe_allow_html=True) # Espaciado final

st.markdown("""
    <style>
    .footer-container {
        width: 100%;
        padding: 40px 0 20px 0;
        margin-top: 60px;
        border-top: 1px solid #27272A;
        text-align: center;
        background-color: transparent;
    }
    .footer-text {
        color: #A1A1AA;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .footer-brand {
        color: #F4F4F5;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }
    .footer-divider {
        width: 50px;
        height: 2px;
        background-color: #00D287;
        margin: 15px auto;
        border-radius: 2px;
    }
    .bi-university {
        color: #00D287;
        margin-right: 8px;
    }
    </style>
    
    <div class="footer-container">
        <div class="footer-brand">
            <i class="bi bi-bank2 bi-university"></i> UNIVERSIDAD MAYOR DE SAN ANDRÉS
        </div>
        <div class="footer-text">
            Facultad de Ciencias Puras y Naturales · Carrera de Informática
        </div>
        <div class="footer-text" style="font-style: italic; opacity: 0.8;">
            Investigación Operativa I · Proyecto de Optimización Cuantitativa
        </div>
        <div class="footer-divider"></div>
        <div class="footer-text" style="font-size: 0.8rem; letter-spacing: 2px;">
            LA PAZ - BOLIVIA · 2026
        </div>
    </div>
""", unsafe_allow_html=True)