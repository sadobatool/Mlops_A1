pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t heart-model .'
            }
        }
        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                    sh "docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD"
                    sh 'docker push your-image-name'
                }
            }
        }
    }
}
