import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# modelo para treino #
emails_treino = [
    "Segue o relatório da semana com os indicadores de produtividade",
    "Vamos marcar uma reunião para discutir melhorias no projeto",
    "Promoção imperdível de sapatos, clique aqui e aproveite",
    "Você ganhou um prêmio, acesse o link para retirar",

]
labels_treino = ["Produtivo", "Produtivo", "Improdutivo", "Improdutivo"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(emails_treino)

modelo = MultinomialNB()
modelo.fit(X_train, labels_treino)


def classificar_email(texto):
    X_test = vectorizer.transform([texto])
    categoria = modelo.predict(X_test)[0]

    if categoria == "Produtivo":
        resposta = "Obrigado pelo envio das informações. Vamos analisar e dar continuidade."
    else:
        resposta = "Este email foi identificado como improdutivo e não requer ação."

    return categoria, resposta


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nome de arquivo inválido"}), 400

    if file and file.filename.endswith(".txt"):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            conteudo = f.read()

        categoria, resposta = classificar_email(conteudo)

        return jsonify({"categoria": categoria, "resposta": resposta})

    return jsonify({"error": "Formato de arquivo inválido. Envie um .txt"}), 400


if __name__ == "__main__":
    app.run(debug=True)
