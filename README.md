# Redact PDF - Ferramenta de Censura e Extração de Dados

Uma ferramenta visual em Python para análise, censura e extração de dados de documentos PDF, com foco em privacidade e proteção de informações sensíveis.

#### ⚠️ Este programa nasceu do mais puro ódio em lidar com arquivos PDF. ⚠️

## 📋 Objetivo

Este programa foi desenvolvido para resolver um problema comum em ambientes corporativos, governamentais e acadêmicos: a necessidade de compartilhar documentos PDF removendo informações confidenciais de forma precisa e confiável. 

O diferencial desta ferramenta está em sua abordagem visual, onde cada elemento de texto é automaticamente identificado e pode ser selecionado individualmente através de cliques, tornando o processo de censura intuitivo e preciso. Além disso, o programa permite a extração estruturada de dados para análise posterior.

## 🎯 Casos de Uso

### Conformidade com Regulamentos de Privacidade
Ideal para adequação à LGPD, GDPR e outras legislações de proteção de dados. Permite remover rapidamente informações pessoais identificáveis antes de compartilhar documentos publicamente.

### Preparação de Documentos Legais
Escritórios de advocacia podem usar a ferramenta para preparar documentos para processos judiciais, removendo informações privilegiadas ou confidenciais antes da fase de descoberta.

### Divulgação de Informações Públicas
Órgãos governamentais podem censurar informações sensíveis em documentos oficiais antes de atender solicitações de acesso à informação, mantendo a transparência sem comprometer a segurança.

### Pesquisa Acadêmica
Pesquisadores podem anonimizar dados em estudos, entrevistas transcritas e relatórios, garantindo a privacidade dos participantes enquanto mantêm a integridade da pesquisa.

### Jornalismo Investigativo
Jornalistas podem proteger suas fontes ao compartilhar documentos vazados, removendo informações que possam identificar a origem do material.

## 🚀 Funcionalidades Principais

- **Visualização inteligente** com detecção automática de todos os elementos de texto
- **Seleção interativa** através de cliques para marcar textos a serem censurados
- **Três modos de censura**: individual, por padrão de texto, ou em massa
- **Exportação de dados** em formato JSON para análise posterior
- **Navegação fluida** com scroll suave e atalhos de teclado
- **Zoom ajustável** para trabalhar com documentos de diferentes tamanhos
- **Interface centralizada** que se adapta ao tamanho da janela

## 💻 Instalação

### Pré-requisitos

O programa requer Python 3.7 ou superior. Certifique-se de ter o Python instalado em seu sistema antes de prosseguir.

### Passo 1: Clone o Repositório

```bash
git clone https://github.com/seu-usuario/pdf-analisador.git
cd pdf-analisador
```

### Passo 2: Crie um Ambiente Virtual (Recomendado)

Criar um ambiente virtual ajuda a manter as dependências do projeto isoladas:

```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instale as Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

```bash
pip install PyMuPDF pillow
```

### Passo 4: Execute o Programa

```bash
python contempt.py
```

## 📖 Como Usar

1. **Abra um PDF**: Use o menu Arquivo → Abrir PDF para carregar seu documento
2. **Selecione textos**: Clique nas caixas vermelhas ao redor dos textos para selecioná-los (ficam azuis)
3. **Escolha a ação**:
   - **Salvar seleção**: Exporta os textos selecionados para análise
   - **Aplicar censura**: Censura apenas os textos selecionados
   - **Censurar ocorrências**: Censura todas as aparições de um texto específico
4. **Salve o resultado**: O programa criará um novo PDF com as censuras aplicadas

## ⚠️ Observações Importantes

A censura aplicada por este programa é permanente e remove completamente o texto do PDF, não apenas o cobre visualmente. Isso garante que as informações censuradas não possam ser recuperadas, mas também significa que você deve sempre manter uma cópia do documento original.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues relatando problemas ou sugerindo melhorias, ou envie pull requests com suas contribuições.

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
