pipeline{
    agent any
    environment{
        dockerImage = ''
        registry = 'amanjain09/pythonapp' 
        registryCredential = 'dockerhub_id'
    }
    stages{
        stage ('Checkout'){
            steps{
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/amanjain639/Team_project']])
            }
        }
        
        stage('Build Docker Image'){
            steps{
                script{
                    dockerImage = docker.build registry
                }
            }
        }
        
        stage('Uploading image'){
            steps{
                script{
                    docker.withRegistry('',registryCredential){
                    dockerImage.push()
                    }
                    
                }
            }
        }
        
    }
}