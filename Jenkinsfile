pipeline{
    agent any

    environment {
        VENV = 'venv'
    }

    stages {
        stage('Cloning Github repo to Jenkins') {
            steps {
                script{
                   echo 'Cloning Github repo to Jenkins...'
                   checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/kar-98k-spec/MLOPS-PROJECT-1.git']])
                }
            }
        }   

        stage('Setting up the virtual environment and installing dependencies') {
            steps {
                script{
                   echo 'Setting up the virtual environment and installing dependencies...'
                   sh '''
                   python -m venv ${VENV_DIR}
                   . ${VENV_DIR}/bin/activate
                   pip install --upgrade pip
                   pip install -e .
                   '''
                }
            }
        }   
    }
}