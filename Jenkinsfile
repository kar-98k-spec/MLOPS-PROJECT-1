pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "mlops-projects-455414"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
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


        stage('Building and pushing Docker image to GCR') {
            steps {
                withCredentials([file(credentialsId : 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script{
                        echo 'Building and pushing Docker image to GCR...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/mlops-project-1:latest .
                        docker push gcr.io/${GCP_PROJECT}/mlops-project-1:latest
                        '''
                    }
                }
            }
        }

        stage('Deploy application to run on Google Cloud') {
            steps {
                withCredentials([file(credentialsId : 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script{
                        echo 'Deploy application to run on Google Cloud...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        
                        gcloud run deploy mlops-project-1 \
                            --image=gcr.io/${GCP_PROJECT}/mlops-project-1:latest \
                            --platform=managed \
                            --region=us-central1 \
                            --allow-unauthenticated \
                            
                        '''
                    }
                }
            }
        }     
    }
}