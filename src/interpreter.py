import ast
import io
import sys
import base64
import matplotlib.pyplot as plt
from IPython.display import display, Image

class LLMCodeInterpreter:
    def __init__(self):
        self.locals = {}
        self.globals = globals().copy()
        self.output = []
        self.figures = []

    def execute(self, code):
        # Reset output and figures
        self.output = []
        self.figures = []

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Execute the code
            exec(compile(tree, "<string>", "exec"), self.globals, self.locals)
            
            # Capture any printed output
            self.output.append(sys.stdout.getvalue())

            # Capture any generated plots
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                self.figures.append(base64.b64encode(buf.getvalue()).decode())
                plt.close(fig)

        except Exception as e:
            self.output.append(f"Error: {str(e)}")

        finally:
            # Restore stdout
            sys.stdout = old_stdout

    def get_output(self):
        return "\n".join(self.output)

    def get_figures(self):
        return self.figures

    def display_results(self):
        print(self.get_output())
        for fig in self.get_figures():
            display(Image(data=base64.b64decode(fig)))

# Example usage:
interpreter = LLMCodeInterpreter()

# LLM-generated code || So we're missing on the code-program parsing bit here, code use ```python ``` to indicate this
llm_code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

print("Sine wave plot generated!")
"""

interpreter.execute(llm_code)
interpreter.display_results()