import os
import sys

import PyPDF2


if __name__ == '__main__':

    pdfFileObj = open("quijote.pdf", 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj, )

    n_pages = len(pdfReader.pages)
    BEGIN_PAGE = 22
    STOP_PAGE = 471
    PATH_WRITE_QUIJOTE1 = os.path.join("./", "..", "databases", "quijote.txt")

    print("NUMBER OF PAGES IN THE PDF: ", n_pages)

    with open(PATH_WRITE_QUIJOTE1, 'w', encoding="utf-8") as wf:

        c = []
        for i in range(BEGIN_PAGE, STOP_PAGE):
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()
            if index := text[:20].find("PRIMERA PARTE") != -1:
                text = text[:index-1] + text[index+len("PRIMERA PARTE"):]
            
            elif index := text[:30].find("SEGUNDA PARTE") != -1:
                i+=3
                continue

            if "Capítulo" in text[:8]:
                split = text.split("\n")
                c.append(split[1])
                text = "\n<|beginchaptername|>\n"+split[0]+"\n"+split[1]+"\n<|endchaptername|>\n\n"+"\n".join(split[2:])

        


            wf.write(text)

            

            # wf.write("\n".join(chunks_processed))

    print("# of chapters: ", len(c))
    print("chapters: ", c)

    #print(pdfReader.pages)
    # pageObj = pdfReader.pages[2]

    # print(pageObj.extract_text())

    pdfFileObj.close()