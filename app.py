import gradio as gr
import colour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import ast
import os

# -------------------------------------------------------------------------
# Core Workflow Functions
# -------------------------------------------------------------------------

def process_uploaded_file(file_obj, data_type, drop_baseline):
    """Processes the raw CSV, saves it to disk, and returns the path (not the DataFrame)
    plus the base file name (without '_processed') for use as a prefix in later tabs.
    If the file is already 'Processed', it is not saved again: the original
    path of the file uploaded by the user is used directly."""
    if file_obj is None:
        return None, None, None, "Please upload a CSV file."

    try:
        data = pd.read_csv(file_obj.name, low_memory=False)
        original_name = os.path.splitext(os.path.basename(file_obj.name))[0]
        # Base name used as a prefix for every output file in later tabs
        base_name = original_name.replace('_processed', '')

        if data_type == "Raw":
            wavelengths = data.iloc[1:, 0]
            reflectances = data.iloc[1:, 1::2]

            master_data = pd.concat([wavelengths, reflectances], axis=1)

            if drop_baseline:
                master_data = master_data.drop(master_data.columns[1], axis=1)

            num_samples = master_data.shape[1] - 1
            master_data.columns = ['Wavelength (nm)'] + [f'Reflectance POS {i} (%R)' for i in range(num_samples)]

            master_data['Wavelength (nm)'] = pd.to_numeric(master_data['Wavelength (nm)'], errors='coerce')
            master_data = master_data.dropna(subset=['Wavelength (nm)'])

            os.makedirs('data-processed', exist_ok=True)
            output_file = os.path.join('data-processed', f'{base_name}_processed.csv')
            master_data.to_csv(output_file, index=False)

            num_samples = master_data.shape[1] - 1
            msg = (
                f"✅ Raw file processed successfully.\n"
                f"- Samples found: {num_samples}\n"
                f"- Wavelength rows: {len(master_data)}\n"
                f"- Saved to: {output_file}"
            )

        else:
            # Already processed: just validate and leave it where it is,
            # without writing it to disk again.
            master_data = data.copy()
            master_data['Wavelength (nm)'] = pd.to_numeric(master_data['Wavelength (nm)'], errors='coerce')
            master_data = master_data.dropna(subset=['Wavelength (nm)'])

            if 'Wavelength (nm)' not in data.columns:
                raise ValueError("The uploaded file does not contain a 'Wavelength (nm)' column.")

            output_file = file_obj.name  # original path of the already-uploaded file
            num_samples = master_data.shape[1] - 1
            msg = (
                f"✅ Processed file loaded (not re-saved).\n"
                f"- Samples found: {num_samples}\n"
                f"- Wavelength rows: {len(master_data)}\n"
                f"- Using original file: {output_file}"
            )

        # Only the PATH (string) and the base name travel between tabs, not the DataFrame
        return output_file, base_name, master_data.head(), msg
    except Exception as e:
        return None, None, None, f"⚠️ Error processing the file: {str(e)}"



def _load_processed_csv(processed_path):
    """Helper: re-reads the processed CSV from disk. Raises if it doesn't exist."""
    if not processed_path:
        raise ValueError("You must process a file in Tab 1 first (no processed file path in memory).")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found on disk: {processed_path}")
    return pd.read_csv(processed_path)


def plot_spectra(processed_path, wl_min, wl_max, base_name):
    """Generates the reflectance spectra plot, reading the processed CSV from disk.
    Returns the figure for display and also the PNG path for download.
    The PNG is saved into results/graphics/reflectance_spectrum/, prefixed
    with the base file name (without '_processed')."""
    try:
        master_data = _load_processed_csv(processed_path)
    except Exception as e:
        gr.Warning(str(e))
        return None, None

    wl_col = 'Wavelength (nm)'
    ref_cols = [c for c in master_data.columns if c != wl_col]
    wavelength = master_data[wl_col].values

    mask = (wavelength >= wl_min) & (wavelength <= wl_max)

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in ref_cols:
        ax.plot(wavelength[mask], master_data[col].values[mask], alpha=0.7)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%R)')
    ax.set_title('Raw Reflectance Spectrum (%R)')
    ax.set_xlim(wl_min, wl_max)
    plt.tight_layout()

    prefix = base_name or "output"
    out_dir = os.path.join('results', 'graphics', 'reflectance_spectrum')
    os.makedirs(out_dir, exist_ok=True)
    spectra_path = os.path.join(out_dir, f'reflectance_spectrum_{prefix}.png')
    fig.savefig(spectra_path, dpi=300, bbox_inches='tight')

    return fig, spectra_path


def _compute_colorimetry_arrays(master_data):
    """
    Shared computation core: given the processed DataFrame, returns
    XYZ, Lab, RGB, RGB_mapped, illuminant_xy and sample_labels.
    Both run_colorimetry and calculate_delta_e call this, each one
    reading the CSV from disk independently.
    """
    data_indexed = master_data.set_index('Wavelength (nm)').sort_index()
    data_normalized = data_indexed / 100

    # 1. Spectral Alignment (MSDS)
    msds = colour.MultiSpectralDistributions(data_normalized)
    shape = colour.SpectralShape(360, 750, 1)
    msds = msds.align(shape)

    # 2. XYZ Tristimulus
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS['D65']
    XYZ = colour.msds_to_XYZ(msds, cmfs, illuminant) / 100

    # 3. CIELab and sRGB
    illuminant_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    Lab = colour.XYZ_to_Lab(XYZ, illuminant=illuminant_xy)
    RGB = colour.XYZ_to_sRGB(XYZ, illuminant=illuminant_xy)

    # Gamut mapping for sRGB
    RGB_mapped = np.clip(RGB, 0, None)
    max_per_sample = RGB_mapped.max(axis=1, keepdims=True)
    scale = np.where(max_per_sample > 1, max_per_sample, 1.0)
    RGB_mapped = RGB_mapped / scale

    sample_labels = [c for c in master_data.columns if c != 'Wavelength (nm)']

    return {
        "msds": msds,
        "XYZ": XYZ,
        "Lab": Lab,
        "RGB": RGB,
        "RGB_mapped": RGB_mapped,
        "illuminant_xy": illuminant_xy,
        "sample_labels": sample_labels,
    }


def export_color_space_data(processed_path, base_name):
    """Builds the MSDS, XYZ, LAB and sRGB tables and saves each one as a
    separate Excel file in results/spaces_values/{XYZ,LAB,MSDS,sRGB}/,
    prefixed with the base file name (without '_processed').
    Returns a preview (head) of each table for display, plus the 4 file paths."""
    try:
        master_data = _load_processed_csv(processed_path)
    except Exception as e:
        gr.Warning(str(e))
        return None, None, None, None, None, None, None, None

    result = _compute_colorimetry_arrays(master_data)
    msds = result["msds"]
    XYZ = result["XYZ"]
    Lab = result["Lab"]
    RGB = result["RGB"]
    sample_labels = result["sample_labels"]

    df_XYZ = pd.DataFrame(XYZ, columns=['X', 'Y', 'Z'], index=sample_labels)
    df_Lab = pd.DataFrame(Lab, columns=['L*', 'a*', 'b*'], index=sample_labels)
    df_sRGB = pd.DataFrame(RGB, columns=['R', 'G', 'B'], index=sample_labels)
    df_msds = pd.DataFrame(msds.values, index=msds.domain, columns=msds.labels)

    prefix = base_name or "output"

    paths = {}
    for sub in ['XYZ', 'LAB', 'MSDS', 'sRGB']:
        out_dir = os.path.join('results', 'spaces_values', sub)
        os.makedirs(out_dir, exist_ok=True)
        paths[sub] = os.path.join(out_dir, f'{prefix}_{sub}.xlsx')

    df_XYZ.to_excel(paths['XYZ'])
    df_Lab.to_excel(paths['LAB'])
    df_msds.to_excel(paths['MSDS'])
    df_sRGB.to_excel(paths['sRGB'])

    # Previews shown in the UI: index turned into a regular column for readability
    preview_XYZ = df_XYZ.reset_index().rename(columns={'index': 'Sample'}).head(10)
    preview_Lab = df_Lab.reset_index().rename(columns={'index': 'Sample'}).head(10)
    preview_sRGB = df_sRGB.reset_index().rename(columns={'index': 'Sample'}).head(10)
    preview_msds = df_msds.reset_index().rename(columns={'index': 'Wavelength (nm)'}).head(10)

    return (
        preview_XYZ, preview_Lab, preview_sRGB, preview_msds,
        paths['XYZ'], paths['LAB'], paths['sRGB'], paths['MSDS'],
    )


def run_colorimetry(processed_path, cols, rows, base_name):
    """Executes CIE calculations, generates visualizations and exports PNGs.
    cols and rows describe the actual physical matrix the samples were taken
    with (e.g. a 10x12 plate), instead of calculating rows automatically.
    PNGs are saved into results/graphics/{XYZ,LAB,sRGB}/, prefixed with the
    base file name (without '_processed')."""
    try:
        master_data = _load_processed_csv(processed_path)
    except Exception as e:
        gr.Warning(str(e))
        return None, None, None, None, None, None

    result = _compute_colorimetry_arrays(master_data)
    XYZ = result["XYZ"]
    RGB_mapped = result["RGB_mapped"]
    illuminant_xy = result["illuminant_xy"]

    prefix = base_name or "output"

    cols = int(cols)
    rows = int(rows)
    n = RGB_mapped.shape[0]
    if rows * cols < n:
        gr.Warning(
            f"The specified matrix ({rows} rows x {cols} columns = {rows*cols} cells) "
            f"is smaller than the number of samples ({n}). Only the first "
            f"{rows*cols} samples will be shown in the grids."
        )

    # --- Visualization 1: Chromaticity Diagram ---
    XYZ_sum = XYZ.sum(axis=1, keepdims=True)
    xy = XYZ[:, :2] / XYZ_sum

    fig_chroma, ax_chroma = plt.subplots(figsize=(6, 7))
    colour.plotting.plot_chromaticity_diagram_CIE1931(axes=ax_chroma, show=False)
    ax_chroma.scatter(xy[:, 0], xy[:, 1], c=RGB_mapped, s=38, edgecolors='white', linewidths=0.5)
    ax_chroma.set_title('CIE 1931 Chromaticity Diagram')
    plt.tight_layout()

    chroma_dir = os.path.join('results', 'graphics', 'XYZ')
    os.makedirs(chroma_dir, exist_ok=True)
    chroma_path = os.path.join(chroma_dir, f'cie1931_xy_{prefix}.png')
    fig_chroma.savefig(chroma_path, dpi=300, bbox_inches='tight')

    # --- Visualization 2: CIELAB Decomposition ---
    cell_size, gap = 1.0, 0.08

    fig_grid, ax_grid = plt.subplots(figsize=(cols * 0.7, rows * 0.7))
    ax_grid.set_xlim(0, cols * cell_size)
    ax_grid.set_ylim(0, rows * cell_size)
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')

    for idx in range(rows * cols):
        row_i = idx // cols
        col_i = idx % cols
        col_mirror = cols - 1 - col_i
        x0 = col_mirror * cell_size + gap / 2
        y0 = (rows - 1 - row_i) * cell_size + gap / 2
        facecolor = RGB_mapped[idx] if idx < n else (1, 1, 1)
        rect = plt.Rectangle((x0, y0), cell_size - gap, cell_size - gap, facecolor=facecolor, edgecolor='white', linewidth=4)
        ax_grid.add_patch(rect)

    buf = io.BytesIO()
    fig_grid.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=120)
    buf.seek(0)
    img_captured = plt.imread(buf)
    img_sRGB = img_captured[..., :3] if img_captured.shape[-1] == 4 else img_captured
    plt.close(fig_grid)

    img_XYZ = colour.sRGB_to_XYZ(img_sRGB)
    img_Lab = colour.XYZ_to_Lab(img_XYZ, illuminant_xy)
    L_ch, a_ch, b_ch = img_Lab[..., 0], img_Lab[..., 1], img_Lab[..., 2]

    cmap_a = LinearSegmentedColormap.from_list('GreenRed', ['green', 'white', 'red'])
    cmap_b = LinearSegmentedColormap.from_list('BlueYellow', ['blue', 'white', 'yellow'])

    fig_decomp, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes[0, 0].imshow(img_sRGB); axes[0, 0].set_title('Reconstructed sRGB Image'); axes[0, 0].axis('off')
    img_L = axes[0, 1].imshow(L_ch, cmap='gray', vmin=0, vmax=100); axes[0, 1].set_title('L* (Luminance)'); axes[0, 1].axis('off'); fig_decomp.colorbar(img_L, ax=axes[0, 1])
    img_a = axes[1, 0].imshow(a_ch, cmap=cmap_a, vmin=-80, vmax=80); axes[1, 0].set_title('a* (Green-Red)'); axes[1, 0].axis('off'); fig_decomp.colorbar(img_a, ax=axes[1, 0])
    img_b = axes[1, 1].imshow(b_ch, cmap=cmap_b, vmin=-80, vmax=80); axes[1, 1].set_title('b* (Blue-Yellow)'); axes[1, 1].axis('off'); fig_decomp.colorbar(img_b, ax=axes[1, 1])
    plt.tight_layout()

    decomp_dir = os.path.join('results', 'graphics', 'LAB')
    os.makedirs(decomp_dir, exist_ok=True)
    decomp_path = os.path.join(decomp_dir, f'cielab_{prefix}.png')
    fig_decomp.savefig(decomp_path, dpi=300, bbox_inches='tight')

    # --- Visualization 3: sRGB Grid with Labels ---
    fig_srgb, ax_srgb = plt.subplots(figsize=(cols * 0.7, rows * 0.7))
    ax_srgb.set_xlim(0, cols * cell_size)
    ax_srgb.set_ylim(0, rows * cell_size)
    ax_srgb.set_aspect('equal')
    ax_srgb.axis('off')

    for idx in range(rows * cols):
        row_i = idx // cols
        col_i = idx % cols
        col_mirror = cols - 1 - col_i
        x0 = col_mirror * cell_size + gap / 2
        y0 = (rows - 1 - row_i) * cell_size + gap / 2
        w = h = cell_size - gap

        if idx < n:
            facecolor = RGB_mapped[idx]
            brightness = 0.299 * facecolor[0] + 0.587 * facecolor[1] + 0.114 * facecolor[2]
            label_colour = 'white' if brightness < 0.5 else 'black'
            ax_srgb.text(x0 + w / 2, y0 + h / 2, str(idx), ha='center', va='center', fontsize=6, color=label_colour)
        else:
            facecolor = (1, 1, 1)

        rect = plt.Rectangle((x0, y0), w, h, facecolor=facecolor, edgecolor='white', linewidth=4)
        ax_srgb.add_patch(rect)
    plt.tight_layout()

    srgb_dir = os.path.join('results', 'graphics', 'sRGB')
    os.makedirs(srgb_dir, exist_ok=True)
    srgb_path = os.path.join(srgb_dir, f'image_sRGB_{prefix}.png')
    fig_srgb.savefig(srgb_path, dpi=300, bbox_inches='tight')

    return fig_chroma, fig_decomp, fig_srgb, chroma_path, decomp_path, srgb_path


def calculate_delta_e(processed_path, group_size, ref_colors_df, base_name):
    """Calculates Delta E00 and exports two separate Excel reports.
    Re-reads the processed CSV from disk and recalculates Lab from scratch,
    instead of depending on an in-memory state from the Colorimetry tab.
    Both Excel files are saved into results/spaces_values/delta_e00/,
    prefixed with the base file name (without '_processed')."""
    try:
        master_data = _load_processed_csv(processed_path)
    except Exception as e:
        gr.Warning(str(e))
        return pd.DataFrame(), pd.DataFrame([{"Error": str(e)}]), None, None

    result = _compute_colorimetry_arrays(master_data)
    Lab = result["Lab"]
    sample_labels = result["sample_labels"]
    illuminant_xy = result["illuminant_xy"]

    ref_dict = {}
    display_labels = {}
    for i, row in ref_colors_df.iterrows():
        name = row.get("Color Name")
        ref_val = row.get("Reference (HEX or [L,a,b])")
        if pd.isna(name) or pd.isna(ref_val) or str(ref_val).strip() == "":
            continue  # empty row (e.g. added when expanding the table but not filled yet)
        color_id = f"Color_{i+1}"
        display_labels[color_id] = name
        ref_dict[color_id] = ref_val

    def parse_reference(ref_val):
        ref_val = str(ref_val).strip()
        if ref_val.startswith('#'):
            rgb = colour.notation.HEX_to_RGB(ref_val)
            xyz = colour.sRGB_to_XYZ(rgb)
            return colour.XYZ_to_Lab(xyz, illuminant=illuminant_xy)
        elif ref_val.startswith('['):
            try:
                parsed = ast.literal_eval(ref_val)
                if len(parsed) == 3:
                    return np.array(parsed)
            except Exception:
                pass
        return np.array([np.nan, np.nan, np.nan])

    detailed_data = []
    for index, label in enumerate(sample_labels):
        group_idx = (index // int(group_size)) + 1
        group_name = f"Color_{group_idx}"

        if group_name not in ref_dict:
            continue

        ref_val = ref_dict[group_name]
        lab_ref = parse_reference(ref_val)
        lab_meas = Lab[index]

        try:
            de00 = colour.difference.delta_E_CIE2000(lab_meas, lab_ref)
        except Exception:
            de00 = np.nan

        detailed_data.append({
            'Color': display_labels.get(group_name, group_name),
            'Position': label,
            'L*': lab_meas[0],
            'a*': lab_meas[1],
            'b*': lab_meas[2],
            'Reference_Input': str(ref_val),
            'L*_ref': lab_ref[0],
            'a*_ref': lab_ref[1],
            'b*_ref': lab_ref[2],
            'ΔE2000': de00
        })

    df_detailed = pd.DataFrame(detailed_data)

    if df_detailed.empty:
        return pd.DataFrame(), pd.DataFrame([{"Message": "No group/color matches found."}]), None, None

    summary_data = []
    for grupo, group_df in df_detailed.groupby('Color', sort=False):
        ref_input = group_df['Reference_Input'].iloc[0]
        summary_data.append({
            'Color': grupo,
            'Reference_Input': ref_input,
            'Mean_ΔE00': group_df['ΔE2000'].mean(),
            'Std_ΔE00': group_df['ΔE2000'].std(),
            'Min_ΔE00': group_df['ΔE2000'].min(),
            'Max_ΔE00': group_df['ΔE2000'].max(),
        })

    df_summary = pd.DataFrame(summary_data)

    # Save two separate Excel files: one for summary, one for detailed data
    prefix = base_name or "output"
    out_dir = os.path.join('results', 'spaces_values', 'delta_e00')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f'Summary_DeltaE_{prefix}.xlsx')
    detailed_path = os.path.join(out_dir, f'DeltaE_Details_{prefix}.xlsx')
    df_summary.to_excel(summary_path, index=False)
    df_detailed.to_excel(detailed_path, index=False)

    return df_summary, df_detailed, summary_path, detailed_path


# -------------------------------------------------------------------------
# Graphical User Interface (Gradio)
# -------------------------------------------------------------------------

default_colors = pd.DataFrame({
    "Color Name": ["Red", "Yellow", "Blue", "Green", "Magenta", "Cyan"],
    "Reference (HEX or [L,a,b])": ["#ee3028", "#fef055", "#0f479a", "#63bc52", "#a666aa", "#8dd3d8"]
})

with gr.Blocks(title="Colorimetry Analysis Hub") as app:
    gr.Markdown("# CIE 1931 Colorimetry Analysis")

    # Only the PATH (string) of the processed CSV is stored, not the DataFrame
    state_processed_path = gr.State()
    # Base file name (without '_processed'), used as a prefix for all output files
    state_base_name = gr.State()

    with gr.Tabs():

        # --- TAB 1: Data Loading ---
        with gr.Tab("1. Data Loading"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload your CSV file")
                    data_type = gr.Radio(["Raw", "Processed"], label="Data Type", value="Raw")
                    drop_baseline = gr.Checkbox(label="Drop Baseline (First column = 100%T)", value=True)
                    btn_process_data = gr.Button("Process Data", variant="primary")

                    status_box = gr.Textbox(label="Status", interactive=False, lines=4)
                    # File export component (manual download of the processed CSV)
                    file_download_csv = gr.File(label="Download Processed CSV File", interactive=False)

                with gr.Column(scale=2):
                    data_preview = gr.Dataframe(label="Processed Data Preview")

            btn_process_data.click(
                fn=process_uploaded_file,
                inputs=[file_input, data_type, drop_baseline],
                outputs=[state_processed_path, state_base_name, data_preview, status_box]
            ).then(
                # Also reflects the processed path in the download component
                fn=lambda p: p,
                inputs=[state_processed_path],
                outputs=[file_download_csv]
            )

        # --- TAB 2: Spectra ---
        with gr.Tab("2. Reflectance Spectra"):
            with gr.Row():
                wl_min = gr.Slider(300, 800, value=360, step=10, label="Minimum Wavelength (nm)")
                wl_max = gr.Slider(300, 800, value=750, step=10, label="Maximum Wavelength (nm)")
                btn_plot_spectra = gr.Button("Plot Spectra")

            spectra_plot = gr.Plot(label="Reflectance Curves")
            download_spectra = gr.File(label="Download Spectrum (PNG)", interactive=False)

            btn_plot_spectra.click(
                fn=plot_spectra,
                inputs=[state_processed_path, wl_min, wl_max, state_base_name],
                outputs=[spectra_plot, download_spectra]
            )

        # --- TAB 3: Color Space Exports (MSDS, XYZ, LAB, sRGB) ---
        with gr.Tab("3. Color Space Data (MSDS, XYZ, LAB, sRGB)"):
            gr.Markdown(
                "Computes the aligned MSDS spectra and the XYZ, CIELAB and sRGB "
                "tristimulus values for every sample, and exports each as a "
                "separate Excel file."
            )
            btn_export_data = gr.Button("Calculate and Export Tables", variant="primary")

            gr.Markdown("**Aligned MSDS Spectra (wavelength x sample)**")
            preview_msds_table = gr.Dataframe(label="MSDS (preview, first 10 rows)")
            download_msds_excel = gr.File(label="Download MSDS (.xlsx)", interactive=False)

            gr.Markdown("**XYZ Tristimulus Values**")
            preview_xyz_table = gr.Dataframe(label="XYZ (preview, first 10 rows)")
            download_xyz_excel = gr.File(label="Download XYZ (.xlsx)", interactive=False)

            gr.Markdown("**CIELAB Values**")
            preview_lab_table = gr.Dataframe(label="LAB (preview, first 10 rows)")
            download_lab_excel = gr.File(label="Download LAB (.xlsx)", interactive=False)

            gr.Markdown("**sRGB Values (raw, pre-gamut-mapping)**")
            preview_srgb_table = gr.Dataframe(label="sRGB (preview, first 10 rows)")
            download_srgb_data_excel = gr.File(label="Download sRGB (.xlsx)", interactive=False)

            btn_export_data.click(
                fn=export_color_space_data,
                inputs=[state_processed_path, state_base_name],
                outputs=[
                    preview_xyz_table, preview_lab_table, preview_srgb_table, preview_msds_table,
                    download_xyz_excel, download_lab_excel, download_srgb_data_excel, download_msds_excel,
                ]
            )

        # --- TAB 4: Colorimetry ---
        with gr.Tab("4. Color Spaces (XYZ, Lab, sRGB)"):
            with gr.Row():
                cols_input = gr.Slider(2, 100, value=6, step=1, label="Columns (matrix width)")
                rows_input = gr.Slider(2, 100, value=6, step=1, label="Rows (matrix height)")
                btn_run_colorimetry = gr.Button("Calculate and Generate Plots", variant="primary")

            gr.Markdown(
                "Specify the **columns x rows** of the physical matrix the "
                "samples were taken with (e.g. a 10x12 plate)."
            )

            with gr.Column():
                plot_chroma = gr.Plot(label="1. CIE 1931 Diagram (XYZ)")
                download_chroma = gr.File(label="Download CIE Diagram (PNG)", interactive=False)

                plot_decomp = gr.Plot(label="2. CIELAB Decomposition (Lab)")
                download_decomp = gr.File(label="Download CIELAB Decomposition (PNG)", interactive=False)

                plot_srgb = gr.Plot(label="3. sRGB Samples")
                download_srgb = gr.File(label="Download sRGB Samples (PNG)", interactive=False)

            btn_run_colorimetry.click(
                fn=run_colorimetry,
                inputs=[state_processed_path, cols_input, rows_input, state_base_name],
                outputs=[plot_chroma, plot_decomp, plot_srgb, download_chroma, download_decomp, download_srgb]
            )

        # --- TAB 5: Delta E ---
        with gr.Tab("5. Delta E00 Analysis"):
            gr.Markdown(
                "This tab re-reads the processed file from Tab 1 and "
                "recalculates colorimetry internally; it does not depend on "
                "having run Tab 4 first."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    group_size = gr.Number(value=6, label="Sample Group Size")
                    ref_colors = gr.Dataframe(
                        value=default_colors,
                        headers=["Color Name", "Reference (HEX or [L,a,b])"],
                        interactive=True,
                        label="Reference Colors",
                        row_count=(6, "dynamic"),
                        column_count=(2, "fixed"),
                    )
                    btn_calc_deltaE = gr.Button("Calculate Differences", variant="primary")

                    # Excel report export components (one file per sheet/table)
                    download_summary_excel = gr.File(label="Download Summary Statistics (.xlsx)", interactive=False)
                    download_detailed_excel = gr.File(label="Download Detailed Data (.xlsx)", interactive=False)

                with gr.Column(scale=2):
                    table_summary = gr.Dataframe(label="Summary by Color")
                    table_detailed = gr.Dataframe(label="Detail by Position")

            btn_calc_deltaE.click(
                fn=calculate_delta_e,
                inputs=[state_processed_path, group_size, ref_colors, state_base_name],
                outputs=[table_summary, table_detailed, download_summary_excel, download_detailed_excel]
            )

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())