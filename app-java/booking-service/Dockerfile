FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /build
COPY pom.xml /build/pom.xml
RUN mvn -q -DskipTests dependency:go-offline --batch-mode
COPY src /build/src
RUN mvn -q -DskipTests package --batch-mode

FROM eclipse-temurin:17-jre
WORKDIR /app
COPY --from=build /build/target/booking-service-0.0.1-SNAPSHOT.jar /app/app.jar
EXPOSE 8083
CMD ["java", "-jar", "/app/app.jar"]