from langchain.text_splitter import CharacterTextSplitter


def get_chunks(raw_text):
    instruction="Represent a sentence for retrieval"
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    data = [[instruction,chunk] for chunk in chunks]
    return chunks,data


if __name__ == '__main__':
    text = """
        Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman serving as the 47th president of the United States since January 2025. A member of the Republican Party, he previously served as the 45th president from 2017 to 2021.

Born in New York City, Trump graduated from the University of Pennsylvania in 1968 with a bachelor's degree in economics. He became president of his family's real estate business in 1971 and oriented it to luxury hotels and casinos. After a series of bankruptcies in the 1990s and 2000s, he began side ventures. From 2004 to 2015, he hosted the reality television show The Apprentice. A political outsider, Trump won the 2016 presidential election against Democratic nominee Hillary Clinton.

In his first term, Trump imposed a travel ban on citizens from six Muslim-majority countries, expanded the U.S.–Mexico border wall, and implemented a brief family separation policy. Domestically, he rolled back environmental and business regulations, signed the Tax Cuts and Jobs Act of 2017, and appointed three Supreme Court justices. In foreign policy, he withdrew the U.S. from agreements on climate, trade, and Iran's nuclear program, began a trade war with China, and met with North Korean leader Kim Jong Un without reaching an agreement on denuclearization. In response to the COVID-19 pandemic, he downplayed its severity, contradicted guidance from public health officials, and enacted the CARES Act stimulus package. Trump was impeached in 2019 for abuse of power and obstruction of Congress, and in 2021 for incitement of insurrection; the Senate acquitted him in both cases. After his first term, scholars and historians ranked him one of the worst presidents in American history.

Trump is the central figure of the Trumpism movement. Many of his comments and actions have been characterized as racially charged, racist or misogynistic, and he has made false and misleading statements and promoted conspiracy theories to a degree unprecedented in American politics. He lost the 2020 presidential election to Joe Biden but refused to concede, falsely claiming electoral fraud, and attempted to overturn the results, including through his involvement in the January 6 Capitol attack in 2021. He founded Trump Media & Technology Group that year. Trump ran again for the 2024 presidential election, defeating incumbent vice president Kamala Harris. He faced legal issues between presidencies and during his 2024 campaign. In civil proceedings, he was found liable for sexual abuse and defamation in 2023, and financial fraud in 2024. He was found guilty of falsifying business records in 2024, making him the first U.S. president convicted of a felony. After his victory in the 2024 presidential election, Trump was sentenced to a penalty-free discharge in 2025, and two other felony indictments against him were dismissed.

Trump began his second presidency by pardoning around 1,500 January 6 rioters, attempting to reduce the size of the federal workforce, and initiating a mass deportation program of illegal immigrants. His broad and extensive use of executive orders has drawn dozens of lawsuits challenging their legality.
    """
    l,data=get_chunks(text)
    print(len(l))
