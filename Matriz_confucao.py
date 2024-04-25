import numpy as np 

class MetricasClassificacao:
    def __init__(self, matriz, nt):
        self.cng_matrix = matriz
        self.nt = nt

    def verifica(self):
        soma_total = np.sum(self.cng_matrix)
        if soma_total == self.nt:
            print("Matriz de confução contem todos as nt possibilidades")
        else:
            print(
                "ERRO, matriz de confução não contem as nt possibilidades, tente novamente")

    def gera_conjuntos_2_2(self):
        num_linhas, num_colunas = self.cng_matrix.shape

        # Inicializando uma lista para armazenar as combinações
        combinações = []

        # Iterando sobre as colunas
        for i in range(num_colunas):
            # Iterando sobre as colunas subsequentes
            for j in range(i + 1, num_colunas):
                # Criando uma nova matriz 2x2 para cada combinação
                nova_matriz = self.cng_matrix[[i, j]][:, [i, j]]
                # Adicionando a nova matriz à lista de combinações
                combinações.append(nova_matriz)
                print(i, j)
                print()
                print(nova_matriz)
        self.combinacoes = combinações

    def calcula_metricas_micro(self):
        matrizes = self.combinacoes
        TP = []
        TN = []
        FN = []
        FP = []
        for temp in matrizes:
            TP.append(temp[0][0])
            TN.append(temp[1][1])
            FN.append(temp[0][1])
            FP.append(temp[1][0])

        TP = np.array(TP).sum()
        TN = np.array(TN).sum()
        FN = np.array(FN).sum()
        FP = np.array(FP).sum()

        ACC = (TP + TN) / (TP+TN+FN+FP)
        MCC = ((TP*TN) - (FP*FN)) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
        SEN = TP/(TP+FN)
        ESP = TN/(TN+FP)
        P = TP/(TP+FP)
        BA = (SEN + ESP)/2
        F1 = (2*P*SEN)/(P+SEN)

        resultados = {
            'ACC': ACC,
            'MCC': MCC,
            'SEN': SEN,
            'ESP': ESP,
            'P': P,
            'BA': BA,
            'F1': F1
        }
        return resultados

    def calcula_metricas_macro(self):
        matrizes = self.combinacoes
        ACC_ = []
        MCC_ = []
        SEN_ = []
        ESP_ = []
        P_ = []
        BA_ = []
        F1_ = []

        for temp in matrizes:
            TP = temp[0][0]
            TN = temp[1][1]
            FN = temp[0][1]
            FP = temp[1][0]

            ACC = (TP + TN) / (TP+TN+FN+FP)
            MCC = ((TP*TN) - (FP*FN)) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
            SEN = TP/(TP+FN)
            ESP = TN/(TN+FP)
            P = TP/(TP+FP)
            BA = (SEN + ESP)/2
            F1 = (2*P*SEN)/(P+SEN)

            ACC_.append(ACC)
            MCC_.append(MCC)
            SEN_.append(SEN)
            ESP_.append(ESP)
            P_.append(P)
            BA_.append(BA)
            F1_.append(F1)

        k = len(matrizes)
        resultados = {
            'ACC': np.array(ACC_).sum()/k,
            'MCC': np.array(MCC_).sum()/k,
            'SEN': np.array(SEN_).sum()/k,
            'ESP': np.array(ESP_).sum()/k,
            'P': np.array(P_).sum()/k,
            'BA': np.array(BA_).sum()/k,
            'F1': np.array(F1_).sum()/k
        }
        return resultados

    def operacional(self):
        self.verifica()
        self.gera_conjuntos_2_2()
        results_micro = self.calcula_metricas_micro()
        results_macro = self.calcula_metricas_macro()
        return results_micro, results_macro