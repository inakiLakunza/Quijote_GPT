import os
import sys

import PyPDF2



if __name__ == '__main__':

    pdfFileObj = open("cads.pdf", 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj, )

    n_pages = len(pdfReader.pages)
    BEGIN_PAGE = 0
    STOP_PAGE = 173
    PATH_WRITE = os.path.join("./", "..", "databases", "cien_anos_de_soledad.txt")

    print("NUMBER OF PAGES IN THE PDF: ", n_pages)

    with open(PATH_WRITE, 'w', encoding="utf-8") as wf:

        for i in range(BEGIN_PAGE, STOP_PAGE):
            
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()

            wf.write(text)

    pdfFileObj.close()