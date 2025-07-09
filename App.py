import gradio as gr
import numpy as np
import tensorflow as tf
from symptoms import SYMPTOMS, DISEASES

# Loading Model File
aiModel = tf.keras.models.load_model('disease_predictor_model.h5')
print(f"Model Loaded successfully")

def predictor(sympArr):
    # If no selected
    if not sympArr:
        return {" Please select symptoms": 1.0}
    
    # vector with 132 symptoms
    inVec = np.zeros(132, dtype=np.float32)
    for symp in sympArr:
        if symp in SYMPTOMS:
            inVec[SYMPTOMS.index(symp)] = 1
    
    # Get All possible diseases
    pred = aiModel.predict(inVec.reshape(1, 132))[0]
    
    # filter top 3 prediction
    top = np.argsort(pred)[-3:][::-1]
    
    # return percentages
    return {
        DISEASES[top[0]]: float(pred[top[0]]),
        DISEASES[top[1]]: float(pred[top[1]]),
        DISEASES[top[2]]: float(pred[top[2]])
    }

# Copy All synptoms
sympAll = SYMPTOMS.copy()

def filter_symp(searched):
    """Filter symptoms based on search term"""
    if not searched.strip():
        return sympAll    # Show All symptoms, you know the default
    else:
        # Show the searched one via search bar
        return [s for s in sympAll if searched.lower() in s.lower()]

with gr.Blocks(title="Disease Predictor", css=".gradio-container {max-width: 1000px}") as app:
    gr.Markdown("# Medical Diagnosis System")
    
    # Store selected symptoms
    stateSel = gr.State([])
    
    with gr.Row():
        search = gr.Textbox(
            placeholder="🔍 Search symptoms...",
            show_label=False,
            elem_id="symptom-search"
        )
    
    with gr.Row():
        with gr.Column():
            symptoms = gr.CheckboxGroup(
                choices=sympAll,
                label="Select Symptoms",
                interactive=True,
                elem_classes=["scrollable-box"],
                elem_id="symptoms-checkbox"
            )
            
            def update_symp_list(searched, sel_itm):
                fil = filter_symp(searched)
                # Only show selected items that are in the filtered list
                visible_selections = [s for s in sel_itm if s in fil]
                return gr.update(choices=fil, value=visible_selections)
            
            def update_stateSel(checkbox_sel, curr_sel):
                
                
                vis_symp = set(symptoms.choices)
                

                current_set = set(curr_sel)
                
                
                checked_set = set(checkbox_sel)
                
                
                unchecked_vis = [s for s in curr_sel 
                                if s in vis_symp and s not in checked_set]
                
                
                updated_sel = [s for s in curr_sel 
                                    if s not in unchecked_vis]
                
                
                for sel in checkbox_sel:
                    if sel not in updated_sel:
                        updated_sel.append(sel)
                
                return updated_sel
            
            
            search.change(
                update_symp_list,
                inputs=[search, stateSel],
                outputs=symptoms
            )
            
            
            symptoms.change(
                update_stateSel,
                inputs=[symptoms, stateSel],
                outputs=stateSel
            )
            
            
            clear_btn = gr.Button("Clear All")
            
            def clear_all():
                return "", gr.update(choices=sympAll, value=[]), []
            
            clear_btn.click(
                clear_all,
                inputs=[],
                outputs=[search, symptoms, stateSel]
            )
        
        with gr.Column():
            output = gr.Label(label="Top Predictions")
            
            
            selected_display = gr.Textbox(
                label="Your Selected Symptoms",
                interactive=False
            )
            
            
            def update_display(selected_items):
                if not selected_items:
                    return "No symptoms selected"
                return ", ".join(selected_items)
            
            
            stateSel.change(
                update_display,
                inputs=stateSel,
                outputs=selected_display
            )
            
            predict_btn = gr.Button("Predict", variant="primary")
            predict_btn.click(
                predictor,
                inputs=stateSel,
                outputs=output
            )

app.css = """
.scrollable-box {
    max-height: 65vh;
    overflow-y: auto;
    padding: 15px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
}
"""

app.launch(share=True)
