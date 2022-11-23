import gradio as gr
import os

def molecule(input_pdb):

    mol = read_mol(input_pdb)

    x = (
        """<!DOCTYPE html>
        <html>
        <head>    
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <style>
    body{
        font-family:sans-serif
    }
    .mol-container {
    width: 100%;
    height: 600px;
    position: relative;
    }
    .mol-container select{
        background-image:None;
    }
    </style>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>  
    <div id="container" class="mol-container"></div>
  
            <script>
               let pdb = `"""
        + mol
        + """`  
      
             $(document).ready(function () {
                let element = $("#container");
                let config = { backgroundColor: "white" };
                let viewer = $3Dmol.createViewer(element, config);
                viewer.addModel(pdb, "pdb");
                viewer.getModel(0).setStyle({}, { cartoon: { color:"spectrum" } });
                viewer.zoomTo();
                viewer.render();
                viewer.zoom(1, 1000); /* slight zoom */
              })
        </script>
        </body></html>"""
    )

    return f"""<iframe style="width: 100%; height: 600px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""

def get_pdb(pdb_code="", filepath=""):
    if pdb_code is None or pdb_code == "":
        try:
            return filepath.name
        except AttributeError as e:
            return None
    else:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"


def read_mol(molpath):
    with open(molpath, "r") as fp:
        lines = fp.readlines()
    mol = ""
    for l in lines:
        mol += l
    return mol



def update(fastaName='',fastaContent=''):
    if(fastaName==''):
      return None
    else:
      return molecule(fastaName+"_pred.pdb")


demo = gr.Blocks()
with demo:
    gr.Markdown("# PP-HelixFold Protein Structure Prediction Demo")
    with gr.Row():
        with gr.Box():
            fastaName = gr.Textbox(interactive=False,label='Fasta label')
            fastaContent = gr.Textbox(interactive=False,label='Fasta content')
            gr.Examples([["T1026", read_mol( "T1026.fasta")],["T1037", read_mol("T1037.fasta")]], [fastaName,fastaContent])
            btn = gr.Button("Predict")
    mol = gr.HTML()
    btn.click(fn=update, inputs=[fastaName,fastaContent], outputs=mol)
demo.launch()
