# CHE1800Y — MEng Project

**Academic credit recorded: Summer 2026 (May–August 2026 session).** The CMTE industry project with Wolseley Express ran January–October 2026; the academic credit session is a separate date field.

## AI checkout for wholesale parts

**Question**  
How can a checkout application identify all parts and quantities in an image while supporting branch staff?

**Result**  
Developed and piloted a computer-vision application covering **107 SKUs**, with **94.5% complete-checkout accuracy on 2,905 validation images**. Every part and quantity must be correct. These figures describe the project validation set, not live production accuracy or a benchmark reproduced here.

**Methods**

- **Machine learning — computer vision:** PyTorch, YOLO, SAM-assisted annotation and DINOv2 features.
- **Software development — desktop:** object-oriented Python, Tkinter/ttk checkout interfaces, PyQt6/Qt annotation tools and worker threads.
- **Data & integration:** SQL preparation and cleaning, HTTP/REST APIs and Azure Blob/Table Storage; an optional FastAPI service supported Dockerized CPU inference.
- **Engineering workflow:** Git, Azure DevOps, Docker, remote Linux/GPU computing, SSH and SCP.
- **Product collaboration:** branch-staff requirements and senior-developer/clerk feedback.

**Confidentiality:** company data, application source code, infrastructure settings and credentials are not shared publicly. This is a public experience summary, not a runnable deployment.
