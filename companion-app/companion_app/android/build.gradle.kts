allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// Redirect build output to a path without spaces to work around Gradle path escaping bug
val safeBuildDir: Directory =
    rootProject.layout.projectDirectory.dir("C:/ATLAS_BUILD")
rootProject.layout.buildDirectory.value(safeBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = safeBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Copy APK to where Flutter expects it
gradle.buildFinished {
    val src = file("C:/ATLAS_BUILD/app/outputs/flutter-apk/app-debug.apk")
    if (src.exists()) {
        val dest = file("../../build/app/outputs/flutter-apk/app-debug.apk")
        dest.parentFile.mkdirs()
        src.copyTo(dest, overwrite = true)
    }
}
tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
